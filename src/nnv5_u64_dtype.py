"""NNv5 7x7 grayscale -> structured 64-bit encoder.

This module defines a hand-authored, block-structured datatype for NNv5 inputs.
It is intentionally *not* a reconstruction-focused image codec. The purpose is to
present reusable, overlapping local propositions that can co-fire and be grouped
by the rule engine with lower novelty pressure than raw bitpacked pixels.

Bit layout (per 16-bit block)
-----------------------------
Each encoded 7x7 image uses 4 overlapping 4x4 blocks. Every block emits 16 bits:

- Bits 0..2  : local floor bucket (3 bits)
- Bits 3..5  : threshold ladder on normalized values (>=T1, >=T2, >=T3)
- Bit 6      : top half brighter than bottom half
- Bit 7      : left half brighter than right half
- Bit 8      : main diagonal stronger than off-diagonal
- Bit 9      : single dominant hotspot exists
- Bit 10     : near-uniform normalized block
- Bit 11     : multiple raised cells (at least 2 cells >= T2)
- Bit 12     : vertical gradient cue present
- Bit 13     : horizontal gradient cue present
- Bit 14     : corner-weighted structure present
- Bit 15     : exceed/saturation detail flag

64-bit word composition
-----------------------
- bits 0..15   = block 0 (rows 0..3, cols 0..3)
- bits 16..31  = block 1 (rows 0..3, cols 3..6)
- bits 32..47  = block 2 (rows 3..6, cols 0..3)
- bits 48..63  = block 3 (rows 3..6, cols 3..6)

The code is structured so the block schema can be reused for future 128-bit
variants by adding more block coordinates while keeping the same 16-bit local
descriptor.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# -----------------------------
# Tunable encoding constants
# -----------------------------
FLOOR_BUCKET_SIZE = 32
THRESH_T1 = 16
THRESH_T2 = 48
THRESH_T3 = 96
EPSILON = 4.0

# Extra cues/guards for relational bits.
HOTSPOT_MIN = 80.0
HOTSPOT_GAP = 24.0
UNIFORM_STD_MAX = 8.0
UNIFORM_RANGE_MAX = 24.0
GRAD_ROW_STEP = 6.0
GRAD_COL_STEP = 6.0
CORNER_ENERGY_MARGIN = 24.0
EXCEED_RANGE = 144.0
EXCEED_STD = 42.0

# 4 overlapping windows over 7x7.
BLOCK_SLICES: tuple[tuple[slice, slice], ...] = (
    (slice(0, 4), slice(0, 4)),
    (slice(0, 4), slice(3, 7)),
    (slice(3, 7), slice(0, 4)),
    (slice(3, 7), slice(3, 7)),
)

BIT_NAMES: dict[int, str] = {
    0: "floor_bucket_bit0",
    1: "floor_bucket_bit1",
    2: "floor_bucket_bit2",
    3: "any_ge_t1",
    4: "any_ge_t2",
    5: "any_ge_t3",
    6: "top_brighter_than_bottom",
    7: "left_brighter_than_right",
    8: "main_diag_stronger_than_off_diag",
    9: "single_dominant_hotspot",
    10: "near_uniform",
    11: "multiple_raised_cells",
    12: "vertical_gradient_present",
    13: "horizontal_gradient_present",
    14: "corner_weighted_structure",
    15: "exceed_saturation",
}


def _to_image_array(image: Any) -> np.ndarray:
    """Validate/coerce input into a 7x7 float32 array in [0, 255]."""
    arr = np.asarray(image)
    if arr.shape != (7, 7):
        raise ValueError(f"Expected shape (7, 7), got {arr.shape}")

    arr = arr.astype(np.float32, copy=False)
    if np.any(arr < 0) or np.any(arr > 255):
        raise ValueError("Input values must be within [0, 255]")
    return arr


def _floor_bucket(floor_val: float) -> int:
    """3-bit coarse floor bucket from local minimum."""
    bucket = int(floor_val) // FLOOR_BUCKET_SIZE
    if bucket < 0:
        return 0
    if bucket > 7:
        return 7
    return bucket


def _extract_block_bits(block: np.ndarray) -> int:
    """Encode one 4x4 block into the 16-bit local descriptor."""
    floor_val = float(np.min(block))
    norm = block - floor_val

    bits = 0

    # Bits 0..2: floor bucket
    floor_bits = _floor_bucket(floor_val)
    bits |= floor_bits

    # Bits 3..5: threshold ladder on normalized values
    if np.any(norm >= THRESH_T1):
        bits |= 1 << 3
    if np.any(norm >= THRESH_T2):
        bits |= 1 << 4
    if np.any(norm >= THRESH_T3):
        bits |= 1 << 5

    # Bit 6: top half brighter than bottom half
    top_mean = float(np.mean(norm[0:2, :]))
    bottom_mean = float(np.mean(norm[2:4, :]))
    if top_mean > bottom_mean + EPSILON:
        bits |= 1 << 6

    # Bit 7: left half brighter than right half
    left_mean = float(np.mean(norm[:, 0:2]))
    right_mean = float(np.mean(norm[:, 2:4]))
    if left_mean > right_mean + EPSILON:
        bits |= 1 << 7

    # Bit 8: main diagonal stronger than off diagonal (2x2 quadrant energy)
    q_tl = float(np.sum(norm[0:2, 0:2]))
    q_tr = float(np.sum(norm[0:2, 2:4]))
    q_bl = float(np.sum(norm[2:4, 0:2]))
    q_br = float(np.sum(norm[2:4, 2:4]))
    if (q_tl + q_br) > (q_tr + q_bl) + EPSILON:
        bits |= 1 << 8

    # Bit 9: single dominant hotspot exists
    flat = np.sort(norm.reshape(-1))
    peak = float(flat[-1])
    second = float(flat[-2])
    if peak >= HOTSPOT_MIN and (peak - second) >= HOTSPOT_GAP:
        bits |= 1 << 9

    # Bit 10: near-uniform after normalization
    norm_std = float(np.std(norm))
    norm_range = float(np.max(norm) - np.min(norm))
    if norm_std <= UNIFORM_STD_MAX and norm_range <= UNIFORM_RANGE_MAX:
        bits |= 1 << 10

    # Bit 11: multiple raised cells present (>=2 cells above T2)
    if int(np.count_nonzero(norm >= THRESH_T2)) >= 2:
        bits |= 1 << 11

    # Bit 12: vertical gradient cue present
    row_means = np.mean(norm, axis=1)
    if (row_means[3] - row_means[0]) >= GRAD_ROW_STEP:
        bits |= 1 << 12

    # Bit 13: horizontal gradient cue present
    col_means = np.mean(norm, axis=0)
    if (col_means[3] - col_means[0]) >= GRAD_COL_STEP:
        bits |= 1 << 13

    # Bit 14: corner-weighted structure present
    corner_sum = float(norm[0, 0] + norm[0, 3] + norm[3, 0] + norm[3, 3])
    interior_sum = float(norm[1, 1] + norm[1, 2] + norm[2, 1] + norm[2, 2])
    if corner_sum > interior_sum + CORNER_ENERGY_MARGIN:
        bits |= 1 << 14

    # Bit 15: exceed/saturation/detail flag
    if norm_range >= EXCEED_RANGE or norm_std >= EXCEED_STD:
        bits |= 1 << 15

    return bits & 0xFFFF


def encode_7x7_to_u64(image: Any) -> int:
    """Encode one 7x7 grayscale sample into a packed 64-bit Python int."""
    arr = _to_image_array(image)

    code = 0
    for block_idx, (row_slice, col_slice) in enumerate(BLOCK_SLICES):
        block = arr[row_slice, col_slice]
        block_code = _extract_block_bits(block)
        code |= block_code << (16 * block_idx)

    return int(code)


def decode_u64_fields(code: int) -> dict[str, Any]:
    """Decode packed code into block-level fields for debugging and inspection."""
    value = int(code) & ((1 << 64) - 1)

    blocks: list[dict[str, Any]] = []
    for block_idx in range(4):
        raw = (value >> (16 * block_idx)) & 0xFFFF
        floor_bucket = raw & 0b111

        bits = {name: bool((raw >> bit) & 1) for bit, name in BIT_NAMES.items() if bit >= 3}

        blocks.append(
            {
                "block_index": block_idx,
                "rows": (BLOCK_SLICES[block_idx][0].start, BLOCK_SLICES[block_idx][0].stop - 1),
                "cols": (BLOCK_SLICES[block_idx][1].start, BLOCK_SLICES[block_idx][1].stop - 1),
                "raw_u16": raw,
                "floor_bucket": floor_bucket,
                "floor_range": (floor_bucket * FLOOR_BUCKET_SIZE, floor_bucket * FLOOR_BUCKET_SIZE + 31),
                "bits": bits,
            }
        )

    return {
        "u64": value,
        "u64_hex": f"0x{value:016x}",
        "blocks": blocks,
        "schema": {
            "thresholds": {"T1": THRESH_T1, "T2": THRESH_T2, "T3": THRESH_T3},
            "epsilon": EPSILON,
            "block_slices": [
                ((rs.start, rs.stop - 1), (cs.start, cs.stop - 1)) for rs, cs in BLOCK_SLICES
            ],
        },
    }


def encode_batch(images: Any) -> np.ndarray:
    """Encode a batch of images into uint64 array of shape (N,)."""
    arr = np.asarray(images)
    if arr.ndim != 3 or arr.shape[1:] != (7, 7):
        raise ValueError(f"Expected batch shape (N, 7, 7), got {arr.shape}")

    out = np.empty(arr.shape[0], dtype=np.uint64)
    for idx in range(arr.shape[0]):
        out[idx] = np.uint64(encode_7x7_to_u64(arr[idx]))
    return out


def demo_self_test() -> dict[str, Any]:
    """Small deterministic helper for quick manual inspection."""
    demo = np.array(
        [
            [0, 0, 8, 16, 16, 8, 0],
            [0, 12, 40, 88, 88, 40, 12],
            [8, 36, 96, 180, 180, 96, 36],
            [16, 88, 180, 255, 255, 180, 88],
            [8, 36, 96, 180, 180, 96, 36],
            [0, 12, 40, 88, 88, 40, 12],
            [0, 0, 8, 16, 16, 8, 0],
        ],
        dtype=np.uint8,
    )
    code = encode_7x7_to_u64(demo)
    return decode_u64_fields(code)


if __name__ == "__main__":
    report = demo_self_test()
    print(report["u64_hex"])
    for block in report["blocks"]:
        print(
            f"block={block['block_index']} raw=0x{block['raw_u16']:04x} "
            f"floor_bucket={block['floor_bucket']}"
        )
