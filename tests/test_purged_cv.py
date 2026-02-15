import numpy as np
import pandas as pd

from validation.cv import PurgedKFold


def test_purged_kfold_no_overlap():
    dates = pd.date_range("2023-01-01", periods=200, freq="h")
    X = pd.DataFrame(np.random.randn(200, 3), index=dates)
    t1 = pd.Series(dates + pd.Timedelta(hours=24), index=dates)

    cv = PurgedKFold(n_splits=5, t1=t1, pct_embargo=0.05)
    for train_idx, test_idx in cv.split(X):
        start = test_idx[0]
        end = test_idx[-1] + 1
        test_start_time = t1.index[start]
        test_end_time = t1.iloc[start:end].max()
        embargo = int(len(X) * cv.pct_embargo)

        for i in train_idx:
            if i < start:
                assert t1.iloc[i] < test_start_time
            elif i >= end:
                assert i >= end + embargo
                assert t1.index[i] > test_end_time


def test_purged_kfold_embargo():
    dates = pd.date_range("2023-01-01", periods=100, freq="h")
    X = pd.DataFrame(np.random.randn(100, 2), index=dates)
    t1 = pd.Series(dates + pd.Timedelta(hours=12), index=dates)

    pct_embargo = 0.1
    embargo = int(len(X) * pct_embargo)
    cv = PurgedKFold(n_splits=5, t1=t1, pct_embargo=pct_embargo)

    for train_idx, test_idx in cv.split(X):
        end = test_idx[-1] + 1
        embargo_range = set(range(end, min(end + embargo, len(X))))
        assert embargo_range.isdisjoint(set(train_idx))
