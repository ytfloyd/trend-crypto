from datetime import datetime, timedelta

import duckdb
import pandas as pd

from scripts.research.run_101_alphas_compute_v0 import load_prices


def test_load_prices_missing_vwap(tmp_path):
    db_path = tmp_path / "test.duckdb"
    con = duckdb.connect(str(db_path))
    con.execute(
        "CREATE TABLE bars_1d_clean(ts TIMESTAMP, symbol VARCHAR, open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE, volume DOUBLE)"
    )
    rows = []
    start = datetime(2024, 1, 1)
    for i in range(3):
        ts = start + timedelta(days=i)
        rows.append((ts, "BTC-USD", 1.0, 1.0, 1.0, 1.0 + i, 100.0))
    con.executemany("INSERT INTO bars_1d_clean VALUES (?, ?, ?, ?, ?, ?, ?)", rows)
    con.close()

    df = load_prices(db_path, "bars_1d_clean", None, None)
    assert "vwap" in df.columns
    assert df["vwap"].notna().all()
    assert (df["vwap"] == df["close"]).all()
