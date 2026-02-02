import pandas as pd

from scripts.research.transtrend_crypto_lib_v2_multispeed import combine_sleeves


def _equity(ts, rets, turns=None, gross=None, danger=None):
    df = pd.DataFrame({"ts": ts, "portfolio_ret": rets})
    df["portfolio_equity"] = (1 + df["portfolio_ret"]).cumprod()
    df["turnover_one_sided"] = turns if turns is not None else 0.0
    df["turnover_two_sided"] = df["turnover_one_sided"] * 2.0
    df["gross_exposure"] = gross if gross is not None else 1.0
    df["danger"] = danger if danger is not None else False
    return df


def test_combine_sleeves_aligns_and_weights_returns():
    ts_fast = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"])
    ts_slow = pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-04"])

    fast = _equity(ts_fast, [0.01, 0.02, -0.01], turns=[0.1, 0.2, 0.3], gross=[0.8, 0.9, 1.0])
    slow = _equity(ts_slow, [0.00, 0.01, 0.02], turns=[0.05, 0.06, 0.07], gross=[0.5, 0.6, 0.7])

    out = combine_sleeves(fast, slow, w_fast=0.3, w_slow=0.7)

    assert list(out["ts"]) == list(pd.to_datetime(["2020-01-02", "2020-01-03"]))

    expected_ret = 0.3 * pd.Series([0.02, -0.01]) + 0.7 * pd.Series([0.00, 0.01])
    pd.testing.assert_series_equal(out["portfolio_ret"].reset_index(drop=True), expected_ret, check_names=False)

    expected_equity = (1 + expected_ret).cumprod()
    pd.testing.assert_series_equal(out["portfolio_equity"].reset_index(drop=True), expected_equity, check_names=False)

    expected_turn = 0.3 * pd.Series([0.2, 0.3]) + 0.7 * pd.Series([0.05, 0.06])
    pd.testing.assert_series_equal(out["turnover_one_sided"].reset_index(drop=True), expected_turn, check_names=False)
