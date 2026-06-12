from __future__ import annotations

import pandas as pd
import polars as pl


def to_alphas_panel(df: pl.DataFrame) -> pl.DataFrame:
    alpha_cols = [c for c in df.columns if c.startswith("alpha_")]
    if "ts" not in df.columns or "symbol" not in df.columns:
        raise ValueError("DataFrame must include 'ts' and 'symbol'.")
    if not alpha_cols:
        raise ValueError("No alpha_* columns found.")
    return df.select(["ts", "symbol"] + alpha_cols).sort(["ts", "symbol"])


def to_pandas_multiindex(df: pl.DataFrame) -> pd.DataFrame:
    panel = to_alphas_panel(df)
    pdf = panel.to_pandas()
    pdf["ts"] = pd.to_datetime(pdf["ts"])
    pdf = pdf.set_index(["symbol", "ts"]).sort_index()
    return pdf
