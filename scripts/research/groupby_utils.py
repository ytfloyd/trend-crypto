import pandas as pd

_PD_MAJOR = int(pd.__version__.split(".")[0])


def apply_by_symbol(df: pd.DataFrame, fn):
    def wrapped(group: pd.DataFrame):
        key = group.name  # capture before any copy
        if _PD_MAJOR >= 3:
            group = group.copy()
            group["symbol"] = key
            group.name = key  # preserve for downstream code
        out = fn(group)
        out = out.copy()
        out["symbol"] = key if _PD_MAJOR >= 3 else group["symbol"].iloc[0]
        return out

    kw = {"include_groups": False} if _PD_MAJOR >= 3 else {}
    return df.groupby("symbol", group_keys=False).apply(wrapped, **kw)


def apply_by_ts(df: pd.DataFrame, fn):
    def wrapped(group: pd.DataFrame):
        key = group.name  # capture before any copy
        if _PD_MAJOR >= 3:
            group = group.copy()
            group["ts"] = key
            group.name = key  # preserve for downstream code
        out = fn(group)
        out = out.copy()
        out["ts"] = key if _PD_MAJOR >= 3 else group["ts"].iloc[0]
        return out

    kw = {"include_groups": False} if _PD_MAJOR >= 3 else {}
    return df.groupby("ts", group_keys=False).apply(wrapped, **kw)
