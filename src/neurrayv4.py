import numpy as np

from neurtypes import Case, State, Diff

class U1XToU1X:
    def __init__(self, match_slot:Case, cases:int, groups: int) -> None:
        assert len(match_slot.shape) == 1, "malformed case"
        assert groups > 0, "group count must be positive"

        self.array_size = cases
        self.array_used = 0
        self.group_size = groups
        self.group_used = 0

        self.match = np.zeros((2, match_slot.shape[0], self.array_size), dtype=match_slot.dtype)
        self.emit = np.zeros((self.array_size, self.group_size), dtype=np.bool_)
        self._pending_diff_groups = np.zeros((0, self.group_size), dtype=np.bool_)

        # Debug stats
        self.debug_case_activations = np.zeros(self.array_size, dtype=np.uint64)
        self.reset_debug_stats()

    # This is forward + reverse passes
    def forward(self, tokens:State) -> Diff:
        assert tokens.dtype == self.match.dtype
        assert tokens.shape[1] == self.match.shape[1]


        match_view = self.match[None, ..., :self.array_used]

        tokens_inv = ~tokens.copy()
        input: Diff = np.stack((tokens, tokens_inv), axis=1)

        # FORWARD
        selection = input[..., None] & match_view

        choices: np.ndarray = (selection == match_view).all((1,2))

        if self.array_used and self.group_used:
            token_groups = (
                choices.astype(np.uint8, copy=False)
                @ self.emit[:self.array_used, :self.group_used].astype(np.uint8, copy=False)
            ) > 0
        else:
            token_groups = np.zeros((tokens.shape[0], self.group_size), dtype=np.bool_)

        if self.array_used:
            case_hits = np.count_nonzero(choices, axis=0).astype(np.uint64, copy=False)
            self.debug_case_activations[:self.array_used] += case_hits
            self.debug_total_case_activations += int(case_hits.sum())

        # REVERSE
        match_broad = np.broadcast_to(match_view, (choices.shape[0], 2, self.match.shape[1], self.array_used))
        reverse: Diff = np.bitwise_or.reduce(match_broad, axis=3, where=choices[:, None, None, :])

        diff: Diff = reverse & input ^ input
        # removing [[0, ..., 0][0, ..., 0]] instances, because harmful to compute
        diff_mask = diff[:, 0].any(axis=1)
        diff_cul: Diff = diff[diff_mask]
        diff_groups = token_groups[diff_mask]

        # dedupe as assign will blindly add multipule indentical cases otherwise
        if diff_cul.shape[0]:
            diff_cul2: Diff
            inverse: np.ndarray
            diff_cul2, inverse = np.unique(diff_cul, axis=0, return_inverse=True)
            pending_groups = np.zeros((diff_cul2.shape[0], self.group_size), dtype=np.bool_)
            if self.group_used:
                np.logical_or.at(
                    pending_groups[:, :self.group_used],
                    inverse,
                    diff_groups[:, :self.group_used],
                )
            self._pending_diff_groups = pending_groups
        else:
            diff_cul2 = diff_cul
            self._pending_diff_groups = np.zeros((0, self.group_size), dtype=np.bool_)

        self.debug_forward_calls += 1
        self.debug_total_inputs += int(tokens.shape[0])
        self.debug_total_diffs += int(diff_cul2.shape[0])

        return diff_cul2

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
        if self.array_used:
            case_activation_mean = float(active_case_activations.mean())

        active_groups = 0
        mean_cases_per_group = 0.0
        if self.array_used and self.group_used:
            active_emit = self.emit[:self.array_used, :self.group_used]
            group_case_counts = np.count_nonzero(active_emit, axis=0)
            active_groups = int(np.count_nonzero(group_case_counts))
            if active_groups:
                mean_cases_per_group = float(group_case_counts[group_case_counts > 0].mean())

        return {
            "forward_calls": self.debug_forward_calls,
            "total_inputs": self.debug_total_inputs,
            "total_diffs": self.debug_total_diffs,
            "total_case_activations": self.debug_total_case_activations,
            "active_cases": self.array_used,
            "avg_diffs_per_forward": avg_diffs_per_forward,
            "avg_diffs_per_input": avg_diffs_per_input,
            "avg_case_activations_per_input": avg_case_activations_per_input,
            "mean_activations_per_case": case_activation_mean,
            "active_groups": active_groups,
            "mean_cases_per_group": mean_cases_per_group,
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
    def assign(self, diff:Diff) -> None:
        assert diff.shape[1] == 2, "diff is malformed"
        assert diff.shape[2] == self.match.shape[1], "case is mismatched"

        if self._pending_diff_groups.shape[0] == diff.shape[0]:
            pending_groups = self._pending_diff_groups.copy()
        else:
            pending_groups = np.zeros((diff.shape[0], self.group_size), dtype=np.bool_)

        # ASSIGN
        diff2 = diff[..., None] ^ self.match[None, ..., :self.array_used]

        diffed_pos = diff2[:, 0].any(axis=1)
        diffed_neg_mask = diffed_pos.all(axis=0)
        new_case_mask = diffed_pos.all(axis=1)
        existing_case_match_mask = ~diffed_pos

        neg_case: np.ndarray = np.bitwise_and.reduce(diff[:, 1], axis=0)

        # APPLY
        self.match[..., :self.array_used][1, :, ~diffed_neg_mask] &= neg_case

        if self.group_used and self.array_used and pending_groups.shape[0]:
            case_group_union = (
                existing_case_match_mask.T.astype(np.uint8, copy=False)
                @ pending_groups[:, :self.group_used].astype(np.uint8, copy=False)
            ) > 0
            self.emit[:self.array_used, :self.group_used] |= case_group_union

        new_case_indices = np.flatnonzero(new_case_mask)
        new_cases = diff[new_case_mask]
        new_case_groups = pending_groups[new_case_mask]

        if new_cases.shape[0]:
            kept = np.ones(new_cases.shape[0], dtype=np.bool_)
            fresh_group_used = False
            for idx, groups in enumerate(new_case_groups):
                source_diff_idx = new_case_indices[idx]

                if groups.any():
                    self.emit[:self.array_used, :self.group_used] |= (
                        existing_case_match_mask[source_diff_idx, :, None]
                        & groups[None, :self.group_used]
                    )
                    continue

                if (not fresh_group_used) and self.group_used < self.group_size:
                    groups[self.group_used] = True
                    self.emit[:self.array_used, self.group_used] |= existing_case_match_mask[source_diff_idx]
                    self.group_used += 1
                    fresh_group_used = True
                else:
                    kept[idx] = False

            new_cases = new_cases[kept]
            new_case_groups = new_case_groups[kept]

        assert self.array_used + new_cases.shape[0] <= self.array_size
        self.match[..., self.array_used:self.array_used+new_cases.shape[0]] = np.permute_dims(new_cases, (1,2,0))
        self.match[1, :, self.array_used:self.array_used+new_cases.shape[0]] &= neg_case[..., None]
        self.emit[self.array_used:self.array_used+new_cases.shape[0]] = new_case_groups

        self.array_used += new_cases.shape[0]
        self._pending_diff_groups = np.zeros((0, self.group_size), dtype=np.bool_)
