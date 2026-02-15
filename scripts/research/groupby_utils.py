import pandas as pd

_PD_MAJOR = int(pd.__version__.split(".")[0])


def apply_by_symbol(df: pd.DataFrame, fn):
    def wrapped(group: pd.DataFrame):
        if _PD_MAJOR >= 3:
            group = group.copy()
            group["symbol"] = group.name
        out = fn(group)
        out = out.copy()
        out["symbol"] = group.name if _PD_MAJOR >= 3 else group["symbol"].iloc[0]
        return out

    kw = {"include_groups": False} if _PD_MAJOR >= 3 else {}
    return df.groupby("symbol", group_keys=False).apply(wrapped, **kw)


def apply_by_ts(df: pd.DataFrame, fn):
    def wrapped(group: pd.DataFrame):
        if _PD_MAJOR >= 3:
            group = group.copy()
            group["ts"] = group.name
        out = fn(group)
        out = out.copy()
        out["ts"] = group.name if _PD_MAJOR >= 3 else group["ts"].iloc[0]
        return out

    kw = {"include_groups": False} if _PD_MAJOR >= 3 else {}
    return df.groupby("ts", group_keys=False).apply(wrapped, **kw)
