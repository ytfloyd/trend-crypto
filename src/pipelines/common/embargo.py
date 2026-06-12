"""CPCV embargo helper shared by the evaluation pipelines."""
from __future__ import annotations


def _apply_embargo(
    blocks: list[list[int]],
    test_combo: tuple[int, ...],
    n_dates: int,
    pct_embargo: float,
) -> tuple[list[int], list[int]]:
    """Build train/test index sets with embargo zones around test blocks.

    Removes `n_embargo` observations from training data on each side of
    every test block to prevent information leakage.
    """
    n_embargo = max(1, int(n_dates * pct_embargo))

    test_idx_set: set[int] = set()
    for g in test_combo:
        test_idx_set.update(blocks[g])

    # Embargo zone: indices within n_embargo of any test index
    embargo_set: set[int] = set()
    for idx in test_idx_set:
        for offset in range(-n_embargo, n_embargo + 1):
            neighbor = idx + offset
            if 0 <= neighbor < n_dates and neighbor not in test_idx_set:
                embargo_set.add(neighbor)

    train_idx = [
        i for i in range(n_dates)
        if i not in test_idx_set and i not in embargo_set
    ]
    test_idx = sorted(test_idx_set)
    return train_idx, test_idx
