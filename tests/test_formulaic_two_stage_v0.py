import polars as pl

from alphas.compiler import compile_formulas
from alphas.factory import apply_warmup_policy


def test_two_stage_rank_delta():
    formulas = [("alpha_001", "rank(delta(close, 1))")]
    plan = compile_formulas(formulas)

    ts = ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]
    df = pl.DataFrame(
        {
            "ts": ts * 2,
            "symbol": ["A"] * 4 + ["B"] * 4,
            "close": [10, 11, 13, 12, 20, 19, 18, 22],
            "volume": [100] * 8,
        }
    ).sort(["symbol", "ts"])

    df = df.with_columns(plan.stage1_exprs)
    df = df.with_columns(plan.stage2_exprs)

    # Stage 1 delta should be per symbol
    ts_col = plan.stage1_exprs[0].meta.output_name()
    a_delta = df.filter(pl.col("symbol") == "A")[ts_col].to_list()
    b_delta = df.filter(pl.col("symbol") == "B")[ts_col].to_list()
    assert a_delta[1] == 1
    assert b_delta[1] == -1

    # Stage 2 rank should be per timestamp (cross-sectional)
    day = df.filter(pl.col("ts") == "2024-01-02")
    vals = day.select(["symbol", "alpha_001"]).sort("symbol").to_dicts()
    # For 2024-01-02, symbol A delta=1, symbol B delta=-1 => A rank > B rank
    assert vals[0]["symbol"] == "A"
    assert vals[1]["symbol"] == "B"
    assert vals[0]["alpha_001"] > vals[1]["alpha_001"]


def test_two_stage_scale_executes_and_is_finite():
    formulas = [("alpha_002", "scale(rank(delta(close, 1)))")]
    plan = compile_formulas(formulas)

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

    series = df.select("alpha_002").to_series()
    assert series.is_finite().all()
