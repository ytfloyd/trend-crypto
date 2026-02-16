from datetime import datetime, timedelta

import duckdb
import polars as pl

from scripts.batch_alpha_tearsheets import (
    load_alpha_panel,
    load_forward_returns,
    merge_alpha_returns,
    compute_gatekeeper_metrics,
    write_survivor_corr,
)


def _make_db(path):
    con = duckdb.connect(str(path))
    con.execute("CREATE TABLE bars_1d_clean(ts TIMESTAMP, symbol VARCHAR, close DOUBLE)")
    start = datetime(2024, 1, 1)
    symbols = [f"S{i}" for i in range(10)]
    rows = []
    for i in range(60):
        ts = start + timedelta(days=i)
        for j, sym in enumerate(symbols):
            # Symbol-dependent drift to induce cross-sectional signal
            close = 100 + j * 10 + (i * (0.1 + j * 0.02))
            rows.append((ts, sym, close))
    con.executemany("INSERT INTO bars_1d_clean VALUES (?, ?, ?)", rows)
    con.close()


def test_gatekeeper_batch(tmp_path):
    db_path = tmp_path / "test.duckdb"
    _make_db(db_path)

    start = datetime(2024, 1, 1)
    symbols = [f"S{i}" for i in range(10)]
    rows = []
    for i in range(60):
        ts = start + timedelta(days=i)
        for j, sym in enumerate(symbols):
            close = 100 + j * 10 + (i * (0.1 + j * 0.02))
            rows.append((ts, sym, close))

    df = pl.DataFrame(
        {
            "ts": [r[0] for r in rows],
            "symbol": [r[1] for r in rows],
        }
    )

    # Construct signals: alpha_pass correlates with returns (symbol index)
    df = df.with_columns(
        [
            pl.col("symbol").str.replace("S", "").cast(pl.Int64).alias("sym_idx"),
        ]
    )
    df = df.with_columns(
        [
            pl.col("sym_idx").cast(pl.Float64).alias("alpha_pass"),
            (pl.col("sym_idx") * 0.9).cast(pl.Float64).alias("alpha_pass2"),
            (-pl.col("sym_idx")).cast(pl.Float64).alias("alpha_fail"),
        ]
    ).drop("sym_idx")

    alpha_path = tmp_path / "alphas.parquet"
    df.write_parquet(alpha_path)

    alpha_df, alpha_cols = load_alpha_panel(str(alpha_path))
    returns = load_forward_returns(str(db_path), "bars_1d_clean", df["ts"].min(), df["ts"].max())
    merged = merge_alpha_returns(alpha_df, returns)

    results = {}
    for alpha in alpha_cols:
        metrics, _ = compute_gatekeeper_metrics(merged, alpha, n_quantiles=5)
        results[alpha] = metrics

    assert results["alpha_pass"].verdict == "PASS"
    assert results["alpha_pass2"].verdict == "PASS"
    assert results["alpha_fail"].verdict == "FAIL"

    survivors = [a for a, m in results.items() if m.verdict == "PASS"]
    corr_path = tmp_path / "survivor_corr.csv"
    write_survivor_corr(merged.select(["ts", "symbol"] + survivors), survivors, corr_path)
    assert corr_path.exists()
