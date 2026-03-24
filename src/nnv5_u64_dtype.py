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


def _to_batch_array(images: Any) -> np.ndarray:
    """Validate/coerce input into a (N, 7, 7) float32 array in [0, 255]."""
    arr = np.asarray(images)
    if arr.ndim != 3 or arr.shape[1:] != (7, 7):
        raise ValueError(f"Expected batch shape (N, 7, 7), got {arr.shape}")
    arr = arr.astype(np.float32, copy=False)
    if np.any(arr < 0) or np.any(arr > 255):
        raise ValueError("Input values must be within [0, 255]")
    return arr


def _extract_overlapping_blocks(batch: np.ndarray) -> np.ndarray:
    """Extract all 4 overlapping 4x4 windows. Output shape: (N, 4, 4, 4)."""
    return np.stack(
        [batch[:, rs, cs] for rs, cs in BLOCK_SLICES],
        axis=1,
    )


def _encode_blocks_to_u16(blocks: np.ndarray) -> np.ndarray:
    """Vectorized encoding of blocks with shape (M, 4, 4) into uint16 descriptors."""
    floor_vals = np.min(blocks, axis=(1, 2))
    norm = blocks - floor_vals[:, None, None]

    bits = np.clip((floor_vals // FLOOR_BUCKET_SIZE).astype(np.int32), 0, 7).astype(np.uint16)

    # Bits 3..5: threshold ladder
    bits |= (np.any(norm >= THRESH_T1, axis=(1, 2)).astype(np.uint16) << np.uint16(3))
    bits |= (np.any(norm >= THRESH_T2, axis=(1, 2)).astype(np.uint16) << np.uint16(4))
    bits |= (np.any(norm >= THRESH_T3, axis=(1, 2)).astype(np.uint16) << np.uint16(5))

    # Bit 6: top brighter than bottom
    top_mean = np.mean(norm[:, 0:2, :], axis=(1, 2))
    bottom_mean = np.mean(norm[:, 2:4, :], axis=(1, 2))
    bits |= ((top_mean > (bottom_mean + EPSILON)).astype(np.uint16) << np.uint16(6))

    # Bit 7: left brighter than right
    left_mean = np.mean(norm[:, :, 0:2], axis=(1, 2))
    right_mean = np.mean(norm[:, :, 2:4], axis=(1, 2))
    bits |= ((left_mean > (right_mean + EPSILON)).astype(np.uint16) << np.uint16(7))

    # Bit 8: main diagonal (quadrants) stronger than off diagonal
    q_tl = np.sum(norm[:, 0:2, 0:2], axis=(1, 2))
    q_tr = np.sum(norm[:, 0:2, 2:4], axis=(1, 2))
    q_bl = np.sum(norm[:, 2:4, 0:2], axis=(1, 2))
    q_br = np.sum(norm[:, 2:4, 2:4], axis=(1, 2))
    bits |= ((((q_tl + q_br) > (q_tr + q_bl + EPSILON)).astype(np.uint16)) << np.uint16(8))

    # Bit 9: single dominant hotspot
    flat_sorted = np.sort(norm.reshape(norm.shape[0], -1), axis=1)
    peak = flat_sorted[:, -1]
    second = flat_sorted[:, -2]
    hotspot = (peak >= HOTSPOT_MIN) & ((peak - second) >= HOTSPOT_GAP)
    bits |= (hotspot.astype(np.uint16) << np.uint16(9))

    # Bit 10: near-uniform
    norm_std = np.std(norm, axis=(1, 2))
    norm_range = np.max(norm, axis=(1, 2)) - np.min(norm, axis=(1, 2))
    uniform = (norm_std <= UNIFORM_STD_MAX) & (norm_range <= UNIFORM_RANGE_MAX)
    bits |= (uniform.astype(np.uint16) << np.uint16(10))

    # Bit 11: multiple raised cells (>=2 at T2)
    raised = np.count_nonzero(norm >= THRESH_T2, axis=(1, 2)) >= 2
    bits |= (raised.astype(np.uint16) << np.uint16(11))

    # Bit 12: vertical gradient
    row_means = np.mean(norm, axis=2)
    vgrad = (row_means[:, 3] - row_means[:, 0]) >= GRAD_ROW_STEP
    bits |= (vgrad.astype(np.uint16) << np.uint16(12))

    # Bit 13: horizontal gradient
    col_means = np.mean(norm, axis=1)
    hgrad = (col_means[:, 3] - col_means[:, 0]) >= GRAD_COL_STEP
    bits |= (hgrad.astype(np.uint16) << np.uint16(13))

    # Bit 14: corner weighted
    corners = norm[:, 0, 0] + norm[:, 0, 3] + norm[:, 3, 0] + norm[:, 3, 3]
    interior = norm[:, 1, 1] + norm[:, 1, 2] + norm[:, 2, 1] + norm[:, 2, 2]
    corner_weighted = corners > (interior + CORNER_ENERGY_MARGIN)
    bits |= (corner_weighted.astype(np.uint16) << np.uint16(14))

    # Bit 15: exceed/saturation flag
    exceed = (norm_range >= EXCEED_RANGE) | (norm_std >= EXCEED_STD)
    bits |= (exceed.astype(np.uint16) << np.uint16(15))

    return bits


def encode_7x7_to_u64(image: Any) -> int:
    """Encode one 7x7 grayscale sample into a packed 64-bit Python int."""
    arr = _to_image_array(image)
    return int(encode_batch(arr[None, ...])[0])


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
    batch = _to_batch_array(images)
    blocks = _extract_overlapping_blocks(batch)                # (N, 4, 4, 4)
    flat_blocks = blocks.reshape(blocks.shape[0] * 4, 4, 4)    # (N*4, 4, 4)

    block_codes = _encode_blocks_to_u16(flat_blocks).reshape(blocks.shape[0], 4).astype(np.uint64)
    shifts = np.array([0, 16, 32, 48], dtype=np.uint64)
    encoded = np.left_shift(block_codes, shifts[None, :])
    return np.sum(encoded, axis=1, dtype=np.uint64)


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
