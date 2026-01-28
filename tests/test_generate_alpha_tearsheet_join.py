from datetime import datetime, timedelta

import duckdb
import polars as pl

from scripts.generate_alpha_tearsheet import build_alpha_tearsheet_frame


def test_build_alpha_tearsheet_frame_join(tmp_path):
    db_path = tmp_path / "test.duckdb"
    con = duckdb.connect(str(db_path))
    con.execute(
        "CREATE TABLE bars_1d_clean(ts TIMESTAMP, symbol VARCHAR, close DOUBLE)"
    )

    rows = []
    start = datetime(2024, 1, 1)
    symbols = ["A", "B"]
    for i in range(5):
        ts = start + timedelta(days=i)
        for j, sym in enumerate(symbols):
            rows.append((ts, sym, 100 + i + j))
    con.executemany("INSERT INTO bars_1d_clean VALUES (?, ?, ?)", rows)
    con.close()

    alpha_df = pl.DataFrame(
        {
            "ts": [r[0] for r in rows],
            "symbol": [r[1] for r in rows],
            "alpha_001": [float(i % 3) for i in range(len(rows))],
        }
    )
    alpha_path = tmp_path / "alphas.parquet"
    alpha_df.write_parquet(alpha_path)

    alpha_panel, prices, merged = build_alpha_tearsheet_frame(
        alphas_path=str(alpha_path),
        alpha_name="alpha_001",
        db_path=str(db_path),
        price_table="bars_1d_clean",
        symbols=None,
    )

    assert not alpha_panel.is_empty()
    assert not prices.is_empty()
    assert not merged.is_empty()
    assert set(["ts", "symbol", "signal", "forward_ret"]).issubset(set(merged.columns))
