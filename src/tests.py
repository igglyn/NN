import argparse
from collections.abc import Callable

import numpy as np

from neurrayv4 import U1XToU1X


def format_debug_stats(neur: U1XToU1X, prefix: str = "debug") -> str:
    stats = neur.debug_snapshot()
    return (
        f"[{prefix}] calls={stats['forward_calls']} "
        f"inputs={stats['total_inputs']} "
        f"diffs={stats['total_diffs']} "
        f"cases={stats['active_cases']} "
        f"total_groups={stats['total_groups']} "
        f"avg_diffs/input={stats['avg_diffs_per_input']:.4f} "
        f"avg_case_acts/input={stats['avg_case_activations_per_input']:.4f} "
        f"mean_acts/case={stats['mean_activations_per_case']:.2f} "
        f"cases/group={stats['mean_cases_in_group']:.2f} "
    )


def print_debug_stats(neur: U1XToU1X, prefix: str = "debug") -> None:
    print(format_debug_stats(neur, prefix=prefix))


def sanity_test():

    eliv = U1XToU1X(np.empty(4, dtype=np.uint8), cases=6, groups=6)

    temp = np.array([[8, 0, 0, 0], [2,0,0,0], [4,0,0,0]], dtype=np.uint8)

    rev, emi, mat = eliv.forward(temp)
    eliv.assign(rev, emi, mat)
    print(eliv.array_used) # 3 cases

    rev, emi, mat = eliv.forward(np.array([[9, 0, 0, 0], [12,0,0,0], [14,0,0,0]], dtype=np.uint8))
    eliv.assign(rev, emi, mat)
    print(eliv.array_used) # 4 cases (adding [1, 0, 0, 0])

    rev, emi, mat = eliv.forward(np.array([[9, 0, 0, 0], [12,0,0,0], [14,0,0,0]], dtype=np.uint8))
    print(eliv.array_used) # 4 cases (It's identical after all)

    rev, emi, mat = eliv.forward(np.array([[16, 0, 0, 0], [8,0,0,0], [4,0,0,0]], dtype=np.uint8))
    print(eliv.array_used) # 4 cases (It's identical after all)





def dataset(
    report_every: int = 1000,
    reporter: Callable[[str], None] | None = print,
    reset_stats_between_stages: bool = True,
):
    import tensorflow_datasets as tfds

    # load full dataset
    mnist, info = tfds.load(
        "mnist",
        split=["train", "test"],
        as_supervised=True,
        with_info=True,
        batch_size=-1,   # <- load everything as big tensors
    )

    # convert to numpy
    train_np, test_np = tfds.as_numpy(mnist)

    # now these are plain numpy arrays
    x_train, y_train = train_np
    x_test, y_test   = test_np

    def chunk_mnist_for_U1X(x: np.ndarray) -> np.ndarray:
        """
        x: numpy array of shape (N, 28, 28)
        returns: (N, 4, 4, 7, 7)
        """
        N, H, W, _ = x.shape
        assert H == 28 and W == 28

        # reshape + transpose trick
        tiles = (
            x
            .reshape(N, 4, 7, 4, 7)
            .transpose(0, 1, 3, 2, 4)
        )

        # flatten 7x7
        tiles = tiles.reshape(tiles.shape[0], 4, 4, -1)    # (N,4,4,49)

        # reorder to tokens
        tiles = tiles.transpose(0, 3, 1, 2)                # (N,49,4,4)

        # now flatten 4x4 to 16 tokens
        tiles = tiles.reshape(tiles.shape[0], tiles.shape[1], -1)  # (N,49,16)

        return tiles

    def recast_u64(a:np.ndarray) -> np.ndarray:
        a = np.ascontiguousarray(a)

        b = a.view(np.uint64)
        b = b.reshape(*a.shape[:-1], 2)
        return b

    # `tiles` is from your chunking function, shape (N, 4, 4, 7, 7)
    tiles_train = chunk_mnist_for_U1X(x_train)
    tiles_eval = chunk_mnist_for_U1X(x_test)

    tiles_train = recast_u64(tiles_train)
    tiles_eval = recast_u64(tiles_eval)

    print(tiles_train.shape)
    gen = np.random.default_rng()
    gen.shuffle(tiles_train, axis=0)

    all_tiles_train = tiles_train.reshape(tiles_train.shape[0] * tiles_train.shape[1], tiles_train.shape[2])


    all_tiles = np.unique(all_tiles_train, axis=0)
    if reporter:
        reporter(f"{all_tiles.shape[0]} total unique tiles")
    # 1972878 tiles as is

    # we love setup being 4 seconds out of 28 second runtime on the poor laptop

    neur = U1XToU1X(np.empty(tiles_train.shape[2], tiles_train.dtype), cases=100_000, groups=40) # case count inflated as chunking code was swapped for 7x7 tiles instead of 4x4

    counter = 0

    for tile in tiles_train:
        if reporter and report_every > 0 and counter % report_every == 0:
            reporter(f"training at: {counter}")
            reporter(format_debug_stats(neur, prefix="train"))
            if counter > 10000:
                break
        rev, emi, mat = neur.forward(tile)
        try:
            neur.assign(rev, emi, mat)
        except AssertionError:
            if reporter:
                reporter(f"Broke at: {counter}, ran out out of cases\n")
            raise
        counter += 1

    if reporter:
        reporter(format_debug_stats(neur, prefix="train-final"))
        reporter(f"\n\n{neur.array_used} cases used!\nMoving into validation\n")
    # 811 as is

    if reset_stats_between_stages:
        neur.reset_debug_stats()

    counter = 0
    misses = 0
    for tile in tiles_eval:
        if reporter and report_every > 0 and counter % report_every == 0:
            reporter(f"eval at: {counter}")
            reporter(format_debug_stats(neur, prefix="eval"))
        rev, _, _ = neur.forward(tile)
        misses += rev.shape[0]
        counter += 1

    if reporter:
        reporter(format_debug_stats(neur, prefix="eval-final"))
        reporter(f"\n\n{misses} inputs were unable to be mapped for eval")
    # 0 as is

    return neur, misses


def main() -> None:
    parser = argparse.ArgumentParser(description="Run NN structural engine diagnostics")
    parser.add_argument("--mode", choices=("dataset", "sanity"), default="dataset")
    parser.add_argument(
        "--report-every",
        type=int,
        default=1000,
        help="Emit periodic debug logs every N batches; set to 0 to disable periodic logs.",
    )
    parser.add_argument(
        "--no-reset-between-stages",
        action="store_true",
        help="Keep debug counters cumulative across training and evaluation.",
    )
    args = parser.parse_args()

    if args.mode == "sanity":
        sanity_test()
        return

    dataset(
        report_every=args.report_every,
        reporter=print,
        reset_stats_between_stages=not args.no_reset_between_stages,
    )


if __name__ == "__main__":
    main()
