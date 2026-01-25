# Purged K-Fold Cross-Validation

Purged K-Fold is designed for time-series labels that overlap in time. It
prevents look-ahead leakage caused by serial correlation in labels (common in
finance). This is the canonical validation primitive for ML/alpha research in
this repo.

## What it solves

Standard random k-fold shuffles observations and leaks future information. In
financial time series, labels often span multiple bars; if a training label
overlaps the test window, the train set contains information from the future
relative to the test set.

Purged K-Fold fixes this by:
- **Purging** any training sample whose label overlaps the test window.
- **Embargoing** a short region immediately after the test window to block
  forward leakage.

## Inputs

- `X`: Feature matrix, indexed by prediction time.
- `t1`: `pd.Series` where:
  - index = prediction time
  - value = label end time
- `pct_embargo`: fraction of total samples to embargo after each test fold.

## Purging

For a given test window, any training sample whose `label_end_time` is greater
than or equal to the **test start time** is removed.

## Embargo

After the test window, a fixed percentage of samples are dropped to prevent
information leakage that occurs immediately after the test period.

## Common failure modes

- **Random k-fold**: breaks temporal ordering and leaks future data.
- **Incorrect t1**: if label end times are wrong or missing, purging is invalid.
- **Index mismatch**: mixing tz-naive and tz-aware timestamps can create silent
  alignment errors.

## Example

```python
import pandas as pd
import numpy as np
from validation.cv import PurgedKFold

dates = pd.date_range("2024-01-01", periods=1000, freq="h")
X = pd.DataFrame(np.random.randn(1000, 5), index=dates)

# 24h label lifetime
# prediction at time T labels until T+24h
# t1 index = prediction time, value = label end time
#
t1 = pd.Series(dates + pd.Timedelta(hours=24), index=dates)

cv = PurgedKFold(n_splits=5, t1=t1, pct_embargo=0.01)
for train_idx, test_idx in cv.split(X):
    pass
```
