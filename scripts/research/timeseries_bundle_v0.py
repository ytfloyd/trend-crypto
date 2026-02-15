import json
import tempfile
from pathlib import Path
from typing import Iterable

import pandas as pd


_BUNDLE_COLUMNS = ["ts", "symbol", "record_type", "field", "value", "meta_json"]


def _can_write_parquet() -> bool:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except Exception:
        return False

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            table = pa.Table.from_pydict({"x": [1.0]})
            pq.write_table(table, str(Path(tmpdir) / "probe.parquet"))
        return True
    except Exception:
        return False


def _to_long(
    df: pd.DataFrame,
    ts_col: str = "ts",
    symbol_col: str = "symbol",
    record_type: str = "FEATURE",
    meta_cols: Iterable[str] | None = None,
) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=_BUNDLE_COLUMNS)

    base = df.copy()
    if ts_col not in base.columns or symbol_col not in base.columns:
        raise ValueError(f"Missing required columns: {ts_col}, {symbol_col}")

    meta_cols = list(meta_cols) if meta_cols else []
    meta_cols = [c for c in meta_cols if c in base.columns]
    meta_json = None
    if meta_cols:
        meta_df = base[meta_cols].copy()
        meta_json = meta_df.apply(
            lambda r: json.dumps(
                {k: (str(v) if isinstance(v, pd.Timestamp) else v) for k, v in r.items() if pd.notna(v)},
                sort_keys=True,
            ),
            axis=1,
        )
        base = base.drop(columns=meta_cols)

    value_cols = [
        c
        for c in base.columns
        if c not in (ts_col, symbol_col)
        and (pd.api.types.is_numeric_dtype(base[c]) or pd.api.types.is_bool_dtype(base[c]))
    ]
    if not value_cols:
        return pd.DataFrame(columns=_BUNDLE_COLUMNS)

    melted = base.melt(
        id_vars=[ts_col, symbol_col],
        value_vars=value_cols,
        var_name="field",
        value_name="value",
        ignore_index=False,
    )
    melted = melted.rename(columns={ts_col: "ts", symbol_col: "symbol"})
    melted["value"] = pd.to_numeric(melted["value"], errors="coerce")
    melted["record_type"] = record_type
    if meta_json is not None:
        melted["meta_json"] = meta_json.loc[melted.index].values
    else:
        melted["meta_json"] = None
    return melted.reset_index(drop=True)[_BUNDLE_COLUMNS]


def write_timeseries_bundle(
    out_dir: str,
    *,
    bars_df: pd.DataFrame | None = None,
    features_df: pd.DataFrame | None = None,
    signals_df: pd.DataFrame | None = None,
    weights_signal_df: pd.DataFrame | None = None,
    weights_held_df: pd.DataFrame | None = None,
    stops_df: pd.DataFrame | None = None,
    trades_df: pd.DataFrame | None = None,
    portfolio_df: pd.DataFrame | None = None,
    write_parquet: bool = False,
    write_csvgz: bool = True,
) -> dict:
    frames = []

    if bars_df is not None:
        frames.append(_to_long(bars_df, record_type="BAR"))
    if features_df is not None:
        frames.append(_to_long(features_df, record_type="FEATURE"))
    if signals_df is not None:
        frames.append(_to_long(signals_df, record_type="SIGNAL"))
    if weights_signal_df is not None:
        frames.append(_to_long(weights_signal_df, record_type="WEIGHT"))
    if weights_held_df is not None:
        frames.append(_to_long(weights_held_df, record_type="WEIGHT"))
    if stops_df is not None:
        frames.append(_to_long(stops_df, record_type="STOP"))
    if trades_df is not None:
        trades_meta = []
        for col in trades_df.columns:
            if col in ("ts", "symbol"):
                continue
            if not (pd.api.types.is_numeric_dtype(trades_df[col]) or pd.api.types.is_bool_dtype(trades_df[col])):
                trades_meta.append(col)
        frames.append(_to_long(trades_df, record_type="TRADE", meta_cols=trades_meta))
    if portfolio_df is not None:
        port = portfolio_df.copy()
        port["symbol"] = "__PORTFOLIO__"
        frames.append(_to_long(port, record_type="PORTFOLIO"))

    bundle = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=_BUNDLE_COLUMNS)
    if not bundle.empty:
        bundle = bundle.sort_values(["ts", "symbol", "record_type", "field"], kind="mergesort")

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    outputs = {"rows": int(len(bundle))}
    if write_parquet:
        if not _can_write_parquet():
            print("[timeseries_bundle_v0] WARNING: parquet unavailable; skipping bundle parquet.")
        else:
            parquet_path = out_path / "timeseries_bundle.parquet"
            try:
                bundle.to_parquet(parquet_path, index=False)
                outputs["parquet"] = str(parquet_path)
            except Exception as exc:
                print(f"[timeseries_bundle_v0] WARNING: parquet write failed; skipping ({exc}).")
    if write_csvgz:
        csvgz_path = out_path / "timeseries_bundle.csv.gz"
        bundle.to_csv(csvgz_path, index=False, compression="gzip")
        outputs["csvgz"] = str(csvgz_path)

    return outputs
