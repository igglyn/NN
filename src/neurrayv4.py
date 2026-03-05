import numpy as np

from neurtypes import Case, State, Diff, Emit

class U1XToU1X:
    def __init__(self, match_slot:Case, cases:int, groups:int) -> None:
        assert len(match_slot.shape) == 1, "malformed case"

        self.array_size = cases
        self.array_used = 0

        self.emit_size = groups
        self.emit_words = (self.emit_size + 63) // 64
        self.emit_used = 0

        self.match = np.zeros((2, match_slot.shape[0], self.array_size), dtype=match_slot.dtype)

        self.emit = np.zeros((2, self.array_size, self.emit_words), dtype=np.uint64)
        self.group_emit_pos = np.zeros((self.emit_size, self.match.shape[1]), dtype=self.match.dtype)
        self.group_emit_neg = np.full(
            (self.emit_size, self.match.shape[1]),
            np.iinfo(self.match.dtype).max,
            dtype=self.match.dtype,
        )

        # Debug stats
        self.debug_case_activations = np.zeros(self.array_size, dtype=np.uint64)
        self.reset_debug_stats()

    def _set_emit_group_bit(self, case_idx: int | slice, group_idx: int, *, is_member: bool) -> None:
        assert 0 <= group_idx < self.emit_size, "emit group index out of bounds"
        word_line, bit_shift = divmod(group_idx, 64)
        group_bit = np.uint64(1) << np.uint64(bit_shift)

        plane = 0 if is_member else 1
        self.emit[plane, case_idx, word_line] |= group_bit

    def _case_pos_mask(self) -> np.ndarray:
        return self.match[0, :, :self.array_used].any(axis=0)

    def _input_neg_observed(self, tokens: State) -> np.ndarray:
        return ~tokens

    def _match_choices(self, tokens: State) -> tuple[np.ndarray, Diff]:
        match_view = self.match[None, ..., :self.array_used]
        tokens_inv = self._input_neg_observed(tokens)
        input_bits: Diff = np.stack((tokens, tokens_inv), axis=1)

        selection = input_bits[..., None] & match_view
        choices: np.ndarray = (selection == match_view).all((1, 2))

        if self.array_used:
            # Groupless safeguard: a case needs at least one positive constraint bit.
            choices &= self._case_pos_mask()[None, :]

        return choices, input_bits

    def _group_ids_for_case(self, case_idx: int) -> list[int]:
        ids: list[int] = []
        for word_idx, word in enumerate(self.emit[0, case_idx]):
            value = int(word)
            while value:
                bit = (value & -value).bit_length() - 1
                group_idx = word_idx * 64 + bit
                if group_idx < self.emit_used:
                    ids.append(group_idx)
                value &= value - 1
        return ids

    def _bit_count_sum(self, values: np.ndarray) -> np.ndarray:
        if hasattr(np, "bit_count"):
            return np.bit_count(values).sum(axis=1, dtype=np.int64)
        unpacked = np.unpackbits(values.view(np.uint8), axis=1)
        return unpacked.sum(axis=1, dtype=np.int64)

    def _case_match_score(self, tokens: State, case_idx: int) -> np.ndarray:
        pos_overlap = self._bit_count_sum(np.bitwise_and(tokens, self.match[0, :, case_idx]))
        neg_overlap = self._bit_count_sum(np.bitwise_and(tokens, self.match[1, :, case_idx]))
        return pos_overlap - neg_overlap

    def _group_emit_for_choices(
        self,
        tokens: State,
        choices: np.ndarray,
        *,
        update_gate: bool,
        confidence_case_count: int,
        match_score_threshold: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        batch = tokens.shape[0]
        pos_emit = np.zeros((batch, self.match.shape[1]), dtype=self.match.dtype)
        neg_emit = np.full((batch, self.match.shape[1]), np.iinfo(self.match.dtype).max, dtype=self.match.dtype)
        matched = choices.any(axis=1)

        for sample_idx in range(batch):
            fired_cases = np.flatnonzero(choices[sample_idx])
            if fired_cases.size == 0:
                neg_emit[sample_idx] = np.zeros(self.match.shape[1], dtype=self.match.dtype)
                continue

            group_case_counts = np.zeros(self.emit_used, dtype=np.uint16)
            group_score = np.full(self.emit_used, -np.iinfo(np.int32).max, dtype=np.int32)
            sample_groups: set[int] = set()

            for case_idx in fired_cases:
                score = int(self._case_match_score(tokens[sample_idx:sample_idx + 1], int(case_idx))[0])
                for group_idx in self._group_ids_for_case(int(case_idx)):
                    sample_groups.add(group_idx)
                    group_case_counts[group_idx] += 1
                    if score > group_score[group_idx]:
                        group_score[group_idx] = score

            if not sample_groups:
                neg_emit[sample_idx] = np.zeros(self.match.shape[1], dtype=self.match.dtype)
                continue

            active_group_idxs = np.fromiter(sample_groups, dtype=np.int64)
            pos_emit[sample_idx] = np.bitwise_or.reduce(self.group_emit_pos[active_group_idxs], axis=0)
            neg_emit[sample_idx] = np.bitwise_and.reduce(self.group_emit_neg[active_group_idxs], axis=0)

            if update_gate:
                observed_pos = tokens[sample_idx]
                observed_neg = self._input_neg_observed(tokens[sample_idx:sample_idx + 1])[0]
                for group_idx in active_group_idxs:
                    if (
                        group_case_counts[group_idx] >= confidence_case_count
                        or group_score[group_idx] >= match_score_threshold
                    ):
                        self.group_emit_pos[group_idx] |= observed_pos
                        self.group_emit_neg[group_idx] &= observed_neg

        final_emit = pos_emit & neg_emit
        return final_emit, matched, choices

    # this is forward and reverse
    def forward(self, tokens:State) -> tuple[Diff, Emit, np.ndarray]:
        assert tokens.dtype == self.match.dtype
        assert tokens.shape[1] == self.match.shape[1]

        match_view = self.match[None, ..., :self.array_used]
        emit_view = self.emit[None, :, :self.array_used]
        choices, input = self._match_choices(tokens)

        emit_mask: np.ndarray = self.emit[:, :self.array_used].any(axis=(0, 2))

        # this one is the merge flag
        was_matched = choices.any(axis=1)
        emit_broad = np.broadcast_to(
            emit_view,
            (choices.shape[0], 2, self.array_used, self.emit_words),
        )
        # we love jank (if nothing matches, it'll be true becasuse the inital true is not compared to anything)
        emit: Emit = np.bitwise_and.reduce(
            emit_broad,
            axis=2,
            where=choices[:, None, :, None] & emit_mask[None, None, :, None],
        )
        emit[~was_matched] = np.uint64(0)

        if self.array_used:
            case_hits = np.count_nonzero(choices, axis=0).astype(np.uint64, copy=False)
            self.debug_case_activations[:self.array_used] += case_hits
            self.debug_total_case_activations += int(case_hits.sum())

        # REVERSE
        match_broad = np.broadcast_to(match_view, (choices.shape[0], 2, self.match.shape[1], self.array_used))
        reverse: Diff = np.bitwise_or.reduce(match_broad, axis=3, where=choices[:, None, None, :])

        diff: Diff = reverse & input ^ input
        # removing [[0, ..., 0][0, ..., 0]] instances, because harmful to compute
        non_zero = diff[:, 0].any(axis=1)
        diff_cul: Diff = diff[non_zero]
        emit_cul: Emit = emit[non_zero]
        was_matched_cul = was_matched[non_zero]

        # dedupe as assign will blindly add multipule indentical cases otherwise
        diff_cul2, idxs = np.unique(diff_cul, axis=0, return_index=True)
        emit_cul2 = emit_cul[idxs]
        was_matched_cul2 = was_matched_cul[idxs]

        self.debug_forward_calls += 1
        self.debug_total_inputs += int(tokens.shape[0])
        self.debug_total_diffs += int(diff_cul2.shape[0])

        return diff_cul2, emit_cul2, was_matched_cul2

    def debug_snapshot(self) -> dict[str, float | int]:
        avg_diffs_per_forward = 0.0
        avg_diffs_per_input = 0.0
        avg_case_activations_per_input = 0.0

        if self.debug_forward_calls:
            avg_diffs_per_forward = self.debug_total_diffs / self.debug_forward_calls
        if self.debug_total_inputs:
            avg_diffs_per_input = self.debug_total_diffs / self.debug_total_inputs
            avg_case_activations_per_input = self.debug_total_case_activations / self.debug_total_inputs

        active_case_activations = self.debug_case_activations[:self.array_used]
        case_activation_mean = 0.0
        group_case_count_mean = 0.0
        group_neg_case_count_mean = 0.0

        if self.array_used:
            case_activation_mean = float(active_case_activations.mean())
        if self.emit_used:
            active_case_emit = self.emit[0, :self.array_used]
            group_case_counts = np.empty(self.emit_used, dtype=np.uint32)
            for group_idx in range(self.emit_used):
                word_idx, bit_idx = divmod(group_idx, 64)
                group_case_counts[group_idx] = np.count_nonzero(
                    ((active_case_emit[:, word_idx] >> np.uint64(bit_idx)) & np.uint64(1))
                )
            group_case_count_mean = float(group_case_counts.mean())
        if self.emit_used:
            active_case_emit = self.emit[1, :self.array_used]
            group_case_counts = np.empty(self.emit_used, dtype=np.uint32)
            for group_idx in range(self.emit_used):
                word_idx, bit_idx = divmod(group_idx, 64)
                group_case_counts[group_idx] = np.count_nonzero(
                    ((active_case_emit[:, word_idx] >> np.uint64(bit_idx)) & np.uint64(1))
                )
            group_neg_case_count_mean = float(group_case_counts.mean())

        return {
            "forward_calls": self.debug_forward_calls,
            "total_inputs": self.debug_total_inputs,
            "total_diffs": self.debug_total_diffs,
            "total_case_activations": self.debug_total_case_activations,
            "total_groups": self.emit_used,
            "active_cases": self.array_used,
            "avg_diffs_per_forward": avg_diffs_per_forward,
            "avg_diffs_per_input": avg_diffs_per_input,
            "avg_case_activations_per_input": avg_case_activations_per_input,
            "mean_activations_per_case": case_activation_mean,
            "mean_cases_in_group": group_case_count_mean,
            "mean_neg_cases_in_group": group_neg_case_count_mean
        }

    def debug_case_activation_counts(self) -> np.ndarray:
        return self.debug_case_activations[:self.array_used].copy()

    def reset_debug_stats(self) -> None:
        self.debug_forward_calls = 0
        self.debug_total_inputs = 0
        self.debug_total_diffs = 0
        self.debug_total_case_activations = 0
        self.debug_case_activations.fill(0)

    # This is assign and apply
    def assign(self, diff:Diff, emit:Emit, was_matched:np.ndarray) -> None:
        assert diff.shape[1] == 2, "diff is malformed"
        assert diff.shape[2] == self.match.shape[1], "case is mismatched"

        # ASSIGN
        diff2 = diff[..., None] ^ self.match[None, ..., :self.array_used]
        diffed_neg_mask = diff2[:, 0].any(axis=1).all(axis=0)
        new_case_mask = diff2[:, 0].any(axis=1).all(axis=1)

        neg_case: np.ndarray = np.bitwise_and.reduce(diff[:, 1], axis=0)

        # APPLY
        self.match[..., :self.array_used][1, :, ~diffed_neg_mask] &= neg_case

        new_cases = diff[new_case_mask & was_matched]

        assert self.array_used + new_cases.shape[0] <= self.array_size
        self.match[..., self.array_used:self.array_used+new_cases.shape[0]] = np.permute_dims(new_cases, (1,2,0))
        self.match[1, :, self.array_used:self.array_used+new_cases.shape[0]] &= neg_case[..., None]

        emit_shift = np.permute_dims(
                    emit[new_case_mask & was_matched],
                    (1, 0, 2),
                )

        self.emit[0, self.array_used:self.array_used+new_cases.shape[0]] |= emit_shift[0]
        self.emit[1, self.array_used:self.array_used+new_cases.shape[0]] &= emit_shift[1]

        self.array_used += new_cases.shape[0]

        new_group_cases = diff[new_case_mask & ~was_matched]

        if new_group_cases.shape[0]:
            assert self.array_used + 1 <= self.array_size
            assert self.emit_used < self.emit_size, "emit group index out of bounds"

            new_group_idx = self.emit_used
            self.emit_used += 1

            self.match[..., self.array_used] = np.permute_dims(new_group_cases[0], (0,1))
            self.match[1, :, self.array_used] &= neg_case

            self._set_emit_group_bit(self.array_used, new_group_idx, is_member=True)
            self.group_emit_pos[new_group_idx] = self.match[0, :, self.array_used]
            self.group_emit_neg[new_group_idx] = self.match[1, :, self.array_used]

            if new_group_idx:
                full_words, rem_bits = divmod(new_group_idx, 64)
                self.emit[1, self.array_used, :full_words] = np.iinfo(np.uint64).max
                if rem_bits:
                    self.emit[1, self.array_used, full_words] |= (np.uint64(1) << np.uint64(rem_bits)) - np.uint64(1)

            self._set_emit_group_bit(slice(None, self.array_used), new_group_idx, is_member=False)
            self.array_used += 1

    def settle_step(
        self,
        state: State,
        *,
        merge_mode: str = "overwrite",
        retain_mask: np.ndarray | None = None,
    ) -> tuple[State, dict[str, int | bool]]:
        choices, _ = self._match_choices(state)
        final_emit, was_matched, _ = self._group_emit_for_choices(
            state,
            choices,
            update_gate=False,
            confidence_case_count=2,
            match_score_threshold=1,
        )

        if merge_mode == "overwrite":
            next_state = final_emit
        elif merge_mode == "leaky":
            if retain_mask is None:
                retain_mask = np.iinfo(state.dtype).max
            next_state = (state & retain_mask) | final_emit
        else:
            raise ValueError("merge_mode must be 'overwrite' or 'leaky'")

        metrics = {
            "matched_inputs": int(was_matched.sum()),
            "unmatched_inputs": int((~was_matched).sum()),
            "active_cases": int(self.array_used),
            "active_groups": int(self.emit_used),
        }
        return next_state, metrics

    def train_emit(
        self,
        state: State,
        *,
        update_gate: bool = True,
        confidence_case_count: int = 2,
        match_score_threshold: int = 1,
    ) -> np.ndarray:
        choices, _ = self._match_choices(state)
        final_emit, _, _ = self._group_emit_for_choices(
            state,
            choices,
            update_gate=update_gate,
            confidence_case_count=confidence_case_count,
            match_score_threshold=match_score_threshold,
        )
        return final_emit
