import polars as pl

from alphas.compiler import compile_formulas
from alphas.factory import apply_warmup_policy


def test_warmup_policy_zeroes_initial_rows():
    formulas = [("alpha_001", "correlation(close, volume, 5)")]
    plan = compile_formulas(formulas)
    assert plan.warmup_bars["alpha_001"] >= 5

    ts = [f"2024-01-{i:02d}" for i in range(1, 11)]
    df = pl.DataFrame(
        {
            "ts": ts * 2,
            "symbol": ["A"] * 10 + ["B"] * 10,
            "close": list(range(10, 20)) + list(range(20, 30)),
            "volume": list(range(100, 110)) + list(range(200, 210)),
        }
    ).sort(["symbol", "ts"])

    df = df.with_columns(plan.stage1_exprs)
    df = df.with_columns(plan.stage2_exprs)
    df, _ = apply_warmup_policy(df, plan.warmup_bars)

    warmup = plan.warmup_bars["alpha_001"]
    for sym in ["A", "B"]:
        sub = df.filter(pl.col("symbol") == sym)
        head_vals = sub.head(warmup).select("alpha_001").to_series().to_list()
        assert all(v == 0.0 for v in head_vals)
        tail_vals = sub.tail(2).select("alpha_001").to_series().to_list()
        assert all(v == v for v in tail_vals)
