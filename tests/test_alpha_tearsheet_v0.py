from pathlib import Path

import polars as pl

from analysis.tearsheet import generate_tearsheet


def test_generate_tearsheet_outputs(tmp_path: Path):
    ts = pl.date_range(start=pl.datetime(2024, 1, 1), end=pl.datetime(2024, 3, 1), interval="1d", eager=True)
    symbols = [f"S{i}" for i in range(10)]
    df = (
        pl.DataFrame({"ts": ts})
        .join(pl.DataFrame({"symbol": symbols}), how="cross")
        .sort(["symbol", "ts"])
    )

    df = df.with_columns(
        [
            (pl.col("ts").rank().over("symbol") % 5).cast(pl.Float64).alias("signal"),
            (pl.col("ts").rank().over("symbol") * 0.001).alias("forward_ret"),
        ]
    )

    output = tmp_path / "alpha_test"
    summary = generate_tearsheet(df, str(output), alpha_name="alpha_test")

    assert (output.with_suffix(".pdf")).exists()
    assert (output.with_suffix(".png")).exists()
    assert (output.with_suffix(".json")).exists()
    assert (output.with_suffix(".pdf")).stat().st_size > 0
    assert (output.with_suffix(".png")).stat().st_size > 0

    assert "mean_ic" in summary
    assert "mean_daily_turnover" in summary
    assert "spread_sharpe" in summary


def test_generate_tearsheet_small_universe(tmp_path: Path):
    ts = pl.date_range(start=pl.datetime(2024, 1, 1), end=pl.datetime(2024, 3, 1), interval="1d", eager=True)
    symbols = ["BTC", "ETH", "SOL"]
    df = (
        pl.DataFrame({"ts": ts})
        .join(pl.DataFrame({"symbol": symbols}), how="cross")
        .sort(["symbol", "ts"])
    )
    df = df.with_columns(
        [
            (pl.col("ts").rank().over("symbol") % 5).cast(pl.Float64).alias("signal"),
            (pl.col("ts").rank().over("symbol") * 0.001).alias("forward_ret"),
        ]
    )
    output = tmp_path / "alpha_small"
    summary = generate_tearsheet(df, str(output), alpha_name="alpha_small", n_quantiles=5)

    assert summary["effective_quantiles"] == 3
    assert (output.with_suffix(".pdf")).exists()
