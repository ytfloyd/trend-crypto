import pandas as pd


def apply_by_symbol(df: pd.DataFrame, fn):
    def wrapped(group: pd.DataFrame):
        out = fn(group)
        out = out.copy()
        out["symbol"] = group.name
        return out

    return df.groupby("symbol", group_keys=False).apply(wrapped)


def apply_by_ts(df: pd.DataFrame, fn):
    def wrapped(group: pd.DataFrame):
        out = fn(group)
        out = out.copy()
        out["ts"] = group.name
        return out

    return df.groupby("ts", group_keys=False).apply(wrapped)
