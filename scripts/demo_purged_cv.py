import numpy as np
import pandas as pd

from validation.cv import PurgedKFold


def _fmt_ts(ts):
    if ts is None:
        return "N/A"
    return ts.isoformat()


def _fmt_hours(val):
    if val is None:
        return "N/A"
    return f"{val:.2f}h"


def main():
    rng = np.random.default_rng(7)
    dates = pd.date_range("2024-01-01", periods=1000, freq="h")
    X = pd.DataFrame(rng.standard_normal((1000, 5)), index=dates)

    # 24h label lifetime: prediction at T labels until T+24h
    t1 = pd.Series(dates + pd.Timedelta(hours=24), index=dates)

    cv = PurgedKFold(n_splits=5, t1=t1, pct_embargo=0.01)

    print("=" * 72)
    print("PurgedKFold Demo — Purge + Embargo Proof")
    print("=" * 72)
    print(f"Dataset: {len(X)} rows | {dates.min()} -> {dates.max()}")
    print("Label lifetime: 24h | pct_embargo: 0.01")
    print("" * 0)

    for i, (train_idx, test_idx) in enumerate(cv.split(X)):
        train_times = t1.index[train_idx]
        test_times = t1.index[test_idx]

        test_start_time = test_times.min()
        test_end_pred_time = test_times.max()
        test_end_label_time = t1.iloc[test_idx].max()

        # Purge proof: check label ends before test start
        train_before_mask = train_times < test_start_time
        if train_before_mask.any():
            max_train_before_label_end = t1.loc[train_times[train_before_mask]].max()
            purge_gap_hours = (
                test_start_time - max_train_before_label_end
            ) / pd.Timedelta(hours=1)
            purge_ok = max_train_before_label_end < test_start_time
        else:
            max_train_before_label_end = None
            purge_gap_hours = None
            purge_ok = True

        # Embargo proof: check first train prediction after test end
        train_after_mask = train_times > test_end_pred_time
        if train_after_mask.any():
            min_train_after_pred_time = train_times[train_after_mask].min()
            embargo_gap_hours = (
                min_train_after_pred_time - test_end_pred_time
            ) / pd.Timedelta(hours=1)
            embargo_ok = embargo_gap_hours > 0
        else:
            min_train_after_pred_time = None
            embargo_gap_hours = None
            embargo_ok = True

        dropped = len(X) - len(train_idx) - len(test_idx)
        purge_status = "✅ PURGE OK" if purge_ok else "❌ PURGE FAIL"
        embargo_status = "✅ EMBARGO OK" if embargo_ok else "❌ EMBARGO FAIL"

        print("-" * 72)
        print(f"Fold {i + 1}")
        print(f"Train: {len(train_idx)} | Test: {len(test_idx)} | Dropped: {dropped}")
        print(f"test_start_time:       {_fmt_ts(test_start_time)}")
        print(f"test_end_pred_time:    {_fmt_ts(test_end_pred_time)}")
        print(f"test_end_label_time:   {_fmt_ts(test_end_label_time)}")
        print(
            "max_train_before_label_end:",
            _fmt_ts(max_train_before_label_end),
            "| purge_gap_hours:",
            _fmt_hours(purge_gap_hours),
        )
        print(
            "min_train_after_pred_time:",
            _fmt_ts(min_train_after_pred_time),
            "| embargo_gap_hours:",
            _fmt_hours(embargo_gap_hours),
        )
        print(f"Status: {purge_status} / {embargo_status}")


if __name__ == "__main__":
    main()
