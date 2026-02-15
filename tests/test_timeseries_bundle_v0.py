import tempfile

import pandas as pd

from scripts.research.timeseries_bundle_v0 import write_timeseries_bundle


def test_timeseries_bundle_writer_csvgz_only():
    bars = pd.DataFrame(
        {
            "ts": ["2020-01-01", "2020-01-01"],
            "symbol": ["AAA", "BBB"],
            "open": [1.0, 2.0],
            "high": [1.1, 2.1],
            "low": [0.9, 1.9],
            "close": [1.0, 2.0],
            "volume": [100.0, 200.0],
        }
    )
    features = pd.DataFrame(
        {"ts": ["2020-01-01"], "symbol": ["AAA"], "ma_5": [1.0]}
    )
    weights_signal = pd.DataFrame(
        {"ts": ["2020-01-01"], "symbol": ["AAA"], "w_signal": [1.0]}
    )
    weights_held = pd.DataFrame(
        {"ts": ["2020-01-01"], "symbol": ["AAA"], "w_held": [0.5]}
    )
    portfolio = pd.DataFrame(
        {"ts": ["2020-01-01"], "portfolio_equity": [1.0], "portfolio_ret": [0.0]}
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        info = write_timeseries_bundle(
            tmpdir,
            bars_df=bars,
            features_df=features,
            weights_signal_df=weights_signal,
            weights_held_df=weights_held,
            portfolio_df=portfolio,
            write_parquet=False,
            write_csvgz=True,
        )
        assert "csvgz" in info

        bundle = pd.read_csv(info["csvgz"])
        assert set(["ts", "symbol", "record_type", "field", "value", "meta_json"]).issubset(bundle.columns)
        assert (bundle["record_type"] == "BAR").any()
        assert (bundle["record_type"] == "PORTFOLIO").any()
