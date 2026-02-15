from __future__ import annotations

from typing import Any

import numpy as np

try:
    from sklearn.model_selection._split import _BaseKFold
except Exception:  # pragma: no cover - fallback when sklearn is unavailable
    class _BaseKFold:  # type: ignore[no-redef]
        def __init__(self, n_splits: int = 5, shuffle: bool = False, random_state: Any = None) -> None:
            self.n_splits = n_splits
            self.shuffle = shuffle
            self.random_state = random_state

        def get_n_splits(self, X: Any = None, y: Any = None, groups: Any = None) -> int:
            return self.n_splits


class PurgedKFold(_BaseKFold):  # type: ignore[misc]
    """
    K-Fold Cross Validation with Purging and Embargo.

    References:
        Marcos López de Prado, Advances in Financial Machine Learning (2018)
        Snippet 7.3, Page 106

    Args:
        n_splits (int): Number of folds.
        t1 (pd.Series): Series with index = prediction_time,
                        value = label_end_time.
        pct_embargo (float): Fraction of samples to embargo after test split.
    """

    def __init__(self, n_splits=5, t1=None, pct_embargo=0.01):
        super().__init__(n_splits=n_splits, shuffle=False, random_state=None)
        if t1 is None:
            raise ValueError("t1 (label end times) must be provided.")
        self.t1 = t1
        self.pct_embargo = pct_embargo

    def split(self, X, y=None, groups=None):
        indices = np.arange(X.shape[0])
        n_samples = X.shape[0]
        embargo = int(n_samples * self.pct_embargo)

        test_ranges = [
            (i[0], i[-1] + 1)
            for i in np.array_split(indices, self.n_splits)
        ]

        for start, end in test_ranges:
            test_indices = indices[start:end]

            test_start_time = self.t1.index[start]
            test_end_time = self.t1.iloc[start:end].max()

            train_indices = []

            for i in indices:
                if i in test_indices:
                    continue

                pred_time = self.t1.index[i]
                label_end = self.t1.iloc[i]

                # Case A: Train sample before test window
                if i < start:
                    if label_end >= test_start_time:
                        continue  # Purge

                # Case B: Train sample after test window
                elif i >= end:
                    if i < end + embargo:
                        continue  # Embargo
                    if pred_time <= test_end_time:
                        continue  # Purge overlap

                train_indices.append(i)

            yield np.array(train_indices), np.array(test_indices)
