#!/usr/bin/env python
"""BTC-only regime-gated EWMA crossover experiment.

This is a small implementation of the design pattern in the RBS note
"Benchmarking the RBS Regime Switching Model to traditional Momentum style
strategies" (2007):

  1. Start with simple always-invested EWMA crossovers.
  2. Add a trend/range regime gate that turns momentum off during choppy
     markets.
  3. Judge success by return quality (Sharpe, drawdown, Calmar), not raw return
     alone.

Signal timing follows the rest of our daily research stack:

  - indicators are known at close[t-1]
  - position is entered at open[t]
  - return is open[t] -> open[t+1]

Transaction costs default to zero to match the RBS benchmarking note.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import duckdb
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_LAKE = "/Users/russellfloyd/Dropbox/NRT/nrt_dev/data/coinbase_crypto_ohlcv_lake.duckdb"
DEFAULT_OUT_DIR = Path("artifacts/research/btc_regime_switching_ma")
ANN = 365.0


@dataclass(frozen=True)
class StrategySpec:
    name: str
    fast: int
    slow: int
    gate: str
    er_window: int = 20
    er_quantile: float = 0.60
    er_exit_quantile: float = 0.40
    er_lookback: int = 252
    vol_fast: int = 5
    vol_slow: int = 20
    vol_floor: float = 0.0
    vol_ceiling: float = math.inf
    hmm_span: int = 5
    hmm_train_days: int = 1460
    hmm_min_train_days: int = 730
    hmm_update_days: int = 7
    hmm_prob_entry: float = 0.55
    hmm_prob_exit: float = 0.45
    hmm_align_with_signal: bool = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BTC-USDC always-on vs regime-gated EWMA crossover experiment."
    )
    parser.add_argument("--lake", default=DEFAULT_LAKE)
    parser.add_argument("--symbol", default="BTC-USDC")
    parser.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--start_date", default="2018-01-01")
    parser.add_argument(
        "--cost_bps",
        type=float,
        default=0.0,
        help="One-way transaction cost in bps, applied to abs(position change).",
    )
    return parser.parse_args()


def load_bars(lake_path: str, symbol: str, start_date: str) -> pd.DataFrame:
    if not Path(lake_path).exists():
        raise FileNotFoundError(f"DuckDB not found: {lake_path}")
    con = duckdb.connect(lake_path, read_only=True)
    try:
        df = con.execute(
            """
            SELECT ts, open, high, low, close, volume
            FROM bars_1d_clean
            WHERE symbol = ?
              AND ts >= CAST(? AS TIMESTAMPTZ)
            ORDER BY ts
            """,
            [symbol, start_date],
        ).df()
    finally:
        con.close()
    if df.empty:
        raise RuntimeError(f"No bars found for symbol={symbol!r}")
    df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_localize(None)
    df = df.set_index("ts").sort_index()
    return df


def efficiency_ratio(close: pd.Series, window: int) -> pd.Series:
    """Kaufman-style trend efficiency: directional move / path length."""
    direction = close.diff(window).abs()
    path = close.diff().abs().rolling(window).sum()
    return direction / path.replace(0.0, np.nan)


def _normal_pdf(x: np.ndarray, means: np.ndarray, sigmas: np.ndarray) -> np.ndarray:
    sigmas = np.maximum(sigmas, 1e-8)
    z = (x[:, None] - means[None, :]) / sigmas[None, :]
    pdf = np.exp(-0.5 * z * z) / (sigmas[None, :] * np.sqrt(2.0 * np.pi))
    return np.maximum(pdf, 1e-300)


def _fit_constrained_hmm3(x: np.ndarray, *, n_iter: int = 25) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit a small 3-state Gaussian HMM ordered as down/range/up.

    This is intentionally lightweight and constrained to the RBS template:
    down mean < 0, range mean = 0, up mean > 0. It is a research gate, not a
    general-purpose HMM package.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 60:
        raise ValueError("Need at least 60 finite observations to fit HMM")

    eps = max(1e-6, float(np.nanstd(x)) * 0.05)
    means = np.array(
        [
            min(float(np.nanquantile(x, 0.25)), -eps),
            0.0,
            max(float(np.nanquantile(x, 0.75)), eps),
        ],
        dtype=float,
    )
    sigmas = np.repeat(max(float(np.nanstd(x)), 1e-5), 3)
    trans = np.full((3, 3), 0.04)
    np.fill_diagonal(trans, 0.92)
    trans = trans / trans.sum(axis=1, keepdims=True)
    pi = np.array([0.25, 0.50, 0.25], dtype=float)

    for _ in range(n_iter):
        b = _normal_pdf(x, means, sigmas)
        n = len(x)
        alpha = np.zeros((n, 3))
        scale = np.zeros(n)
        alpha[0] = pi * b[0]
        scale[0] = alpha[0].sum()
        alpha[0] /= scale[0]
        for t in range(1, n):
            alpha[t] = (alpha[t - 1] @ trans) * b[t]
            scale[t] = alpha[t].sum()
            if scale[t] <= 0 or not np.isfinite(scale[t]):
                scale[t] = 1e-300
            alpha[t] /= scale[t]

        beta = np.ones((n, 3))
        for t in range(n - 2, -1, -1):
            beta[t] = trans @ (b[t + 1] * beta[t + 1])
            beta[t] /= max(scale[t + 1], 1e-300)

        gamma = alpha * beta
        gamma = gamma / np.maximum(gamma.sum(axis=1, keepdims=True), 1e-300)

        xi_sum = np.full((3, 3), 1e-3)
        for t in range(n - 1):
            xi = alpha[t, :, None] * trans * (b[t + 1] * beta[t + 1])[None, :]
            xi_sum += xi / max(float(xi.sum()), 1e-300)
        trans = xi_sum / xi_sum.sum(axis=1, keepdims=True)
        pi = 0.95 * gamma[0] + 0.05 * np.array([0.25, 0.50, 0.25])
        pi = pi / pi.sum()

        weights = gamma.sum(axis=0)
        raw_means = (gamma * x[:, None]).sum(axis=0) / np.maximum(weights, 1e-12)
        means = np.array(
            [
                min(float(raw_means[0]), -eps),
                0.0,
                max(float(raw_means[2]), eps),
            ],
            dtype=float,
        )
        var = (gamma * (x[:, None] - means[None, :]) ** 2).sum(axis=0) / np.maximum(weights, 1e-12)
        sigmas = np.sqrt(np.maximum(var, 1e-8))

    return means, sigmas, trans, pi


def _filter_hmm_prob(prev_prob: np.ndarray, x_t: float, means: np.ndarray, sigmas: np.ndarray, trans: np.ndarray) -> np.ndarray:
    pred = prev_prob @ trans
    b = _normal_pdf(np.array([x_t]), means, sigmas)[0]
    prob = pred * b
    denom = prob.sum()
    if denom <= 0 or not np.isfinite(denom):
        return np.array([0.0, 1.0, 0.0])
    return prob / denom


def hmm3_state_probabilities(close: pd.Series, spec: StrategySpec) -> pd.DataFrame:
    """Point-in-time weekly-updated up/range/down probabilities.

    Input follows the RBS EM note: a 5-day EWMA of daily log returns. The model is
    recalibrated every `hmm_update_days` observations using a trailing training
    window, then filtered daily with the last fitted parameters.
    """
    feature = np.log(close).diff().ewm(span=spec.hmm_span, adjust=False).mean()
    probs = pd.DataFrame(
        {
            "p_down": 0.0,
            "p_range": 1.0,
            "p_up": 0.0,
        },
        index=close.index,
        dtype=float,
    )

    means: np.ndarray | None = None
    sigmas: np.ndarray | None = None
    trans: np.ndarray | None = None
    prob = np.array([0.0, 1.0, 0.0], dtype=float)
    last_fit_i = -10_000
    finite_feature = feature.dropna()

    for i, ts in enumerate(feature.index):
        x_t = feature.loc[ts]
        if not np.isfinite(x_t):
            continue

        hist = finite_feature.loc[:ts].tail(spec.hmm_train_days)
        if len(hist) >= spec.hmm_min_train_days and (i - last_fit_i) >= spec.hmm_update_days:
            try:
                means, sigmas, trans, fitted_pi = _fit_constrained_hmm3(hist.to_numpy())
                # Use the fitted model's filtered probability at the end of the
                # training window as the live prior for today's state.
                prob = fitted_pi.copy()
                for x_hist in hist.to_numpy():
                    prob = _filter_hmm_prob(prob, float(x_hist), means, sigmas, trans)
                last_fit_i = i
            except ValueError:
                pass

        if means is not None and sigmas is not None and trans is not None:
            prob = _filter_hmm_prob(prob, float(x_t), means, sigmas, trans)
            probs.loc[ts, ["p_down", "p_range", "p_up"]] = prob

    return probs


def build_regime_gate(close: pd.Series, spec: StrategySpec) -> pd.Series:
    """Return a point-in-time trend-on boolean series."""
    if spec.gate == "always_on":
        return pd.Series(True, index=close.index)

    if spec.gate in {"hmm3_weekly", "hmm3_weekly_sticky"}:
        probs = hmm3_state_probabilities(close, spec)
        trend_prob = probs[["p_down", "p_up"]].max(axis=1)
        if spec.gate == "hmm3_weekly":
            return (trend_prob >= spec.hmm_prob_entry).fillna(False)

        state = []
        is_on = False
        for ts in close.index:
            p = float(trend_prob.loc[ts])
            if not is_on and p >= spec.hmm_prob_entry:
                is_on = True
            elif is_on and p < spec.hmm_prob_exit:
                is_on = False
            state.append(is_on)
        return pd.Series(state, index=close.index)

    log_ret = np.log(close).diff()
    er = efficiency_ratio(close, spec.er_window)
    er_threshold = (
        er.rolling(spec.er_lookback, min_periods=max(60, spec.er_window * 3))
        .quantile(spec.er_quantile)
        .shift(1)
    )
    er_on = er >= er_threshold

    if spec.gate == "er":
        return er_on.fillna(False)

    vol_on = pd.Series(True, index=close.index)
    if spec.gate in {"er_vol", "er_hyst_vol"}:
        vol_fast = log_ret.rolling(spec.vol_fast).std()
        vol_slow = log_ret.rolling(spec.vol_slow).std()
        vol_ratio = vol_fast / vol_slow.replace(0.0, np.nan)
        vol_on = (vol_ratio >= spec.vol_floor) & (vol_ratio <= spec.vol_ceiling)

    if spec.gate == "er_vol":
        return (er_on & vol_on).fillna(False)

    if spec.gate in {"er_hyst", "er_hyst_vol"}:
        er_exit = (
            er.rolling(spec.er_lookback, min_periods=max(60, spec.er_window * 3))
            .quantile(spec.er_exit_quantile)
            .shift(1)
        )
        state = []
        is_on = False
        for ts in close.index:
            enter_ready = bool(er.loc[ts] >= er_threshold.loc[ts]) if pd.notna(er_threshold.loc[ts]) else False
            exit_ready = bool(er.loc[ts] < er_exit.loc[ts]) if pd.notna(er_exit.loc[ts]) else True
            vol_ready = bool(vol_on.loc[ts]) if pd.notna(vol_on.loc[ts]) else False
            if not is_on and enter_ready and vol_ready:
                is_on = True
            elif is_on and (exit_ready or not vol_ready):
                is_on = False
            state.append(is_on)
        return pd.Series(state, index=close.index)

    raise ValueError(f"Unknown gate: {spec.gate}")


def ewma_signal(close: pd.Series, fast: int, slow: int) -> pd.Series:
    fast_ewm = close.ewm(span=fast, adjust=False, min_periods=fast).mean()
    slow_ewm = close.ewm(span=slow, adjust=False, min_periods=slow).mean()
    sign = np.sign(fast_ewm - slow_ewm)
    return sign.replace(0.0, np.nan).ffill().fillna(0.0)


def run_strategy(bars: pd.DataFrame, spec: StrategySpec, cost_bps: float) -> pd.DataFrame:
    close = bars["close"]
    signal = ewma_signal(close, spec.fast, spec.slow)
    if spec.gate in {"hmm3_weekly", "hmm3_weekly_sticky"}:
        hmm_probs = hmm3_state_probabilities(close, spec)
        if spec.hmm_align_with_signal:
            aligned_gate = (
                ((signal > 0) & (hmm_probs["p_up"] >= spec.hmm_prob_entry))
                | ((signal < 0) & (hmm_probs["p_down"] >= spec.hmm_prob_entry))
            )
            if spec.gate == "hmm3_weekly_sticky":
                state = []
                is_on = False
                for ts in close.index:
                    trend_prob = (
                        hmm_probs.loc[ts, "p_up"] if signal.loc[ts] > 0
                        else hmm_probs.loc[ts, "p_down"] if signal.loc[ts] < 0
                        else 0.0
                    )
                    if not is_on and trend_prob >= spec.hmm_prob_entry:
                        is_on = True
                    elif is_on and trend_prob < spec.hmm_prob_exit:
                        is_on = False
                    state.append(is_on)
                gate = pd.Series(state, index=close.index)
            else:
                gate = aligned_gate.fillna(False)
        else:
            trend_prob = hmm_probs[["p_down", "p_up"]].max(axis=1)
            gate = (trend_prob >= spec.hmm_prob_entry).fillna(False)
    else:
        gate = build_regime_gate(close, spec)
        hmm_probs = None
    gated_signal = signal.where(gate, 0.0)

    # Signal at close[t-1], enter at open[t], earn open[t] -> open[t+1].
    position = gated_signal.shift(1).fillna(0.0)
    gross_ret = bars["open"].shift(-1) / bars["open"] - 1.0
    turnover = position.diff().abs().fillna(position.abs())
    cost = turnover * (cost_bps / 10_000.0)
    strategy_ret = (position * gross_ret - cost).fillna(0.0)

    out = pd.DataFrame(
        {
            "open": bars["open"],
            "close": close,
            "signal": signal,
            "regime_on": gate.astype(float),
            "position": position,
            "gross_ret": gross_ret,
            "strategy_ret": strategy_ret,
            "turnover": turnover,
        },
        index=bars.index,
    )
    if hmm_probs is not None:
        out = out.join(hmm_probs)
    out["equity"] = (1.0 + out["strategy_ret"]).cumprod()
    return out


def metrics(result: pd.DataFrame) -> dict[str, float | int | str]:
    rets = result["strategy_ret"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    equity = (1.0 + rets).cumprod()
    nonzero = rets[rets != 0]
    if len(equity) < 2:
        years = 0.0
    else:
        years = (equity.index[-1] - equity.index[0]).days / 365.25
    total_return = float(equity.iloc[-1] - 1.0)
    cagr = float(equity.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 else 0.0
    vol = float(rets.std(ddof=0) * np.sqrt(ANN))
    sharpe = float(rets.mean() / rets.std(ddof=0) * np.sqrt(ANN)) if rets.std(ddof=0) > 0 else 0.0
    peak = equity.cummax()
    dd = equity / peak - 1.0
    max_dd = float(dd.min())
    calmar = float(cagr / abs(max_dd)) if max_dd < 0 else 0.0
    trade_count = int((result["position"].diff().abs().fillna(0.0) > 0).sum())
    exposure = float(result["position"].abs().mean())
    regime_on = float(result["regime_on"].mean())
    hit_rate = float((nonzero > 0).mean()) if len(nonzero) else 0.0
    trough_ts = str(dd.idxmin())
    recovery_ts = ""
    if max_dd < 0:
        post = equity.loc[dd.idxmin() :]
        recovered = post[post >= peak.loc[dd.idxmin()]]
        if len(recovered):
            recovery_ts = str(recovered.index[0])
    return {
        "start": str(equity.index[0]),
        "end": str(equity.index[-1]),
        "years": years,
        "total_return": total_return,
        "cagr": cagr,
        "ann_vol": vol,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "calmar": calmar,
        "trade_count": trade_count,
        "avg_exposure": exposure,
        "pct_regime_on": regime_on,
        "hit_rate_active_days": hit_rate,
        "trough_ts": trough_ts,
        "recovery_ts": recovery_ts,
    }


def default_specs() -> list[StrategySpec]:
    specs: list[StrategySpec] = []
    for fast, slow in [(10, 20), (20, 50)]:
        specs.append(
            StrategySpec(
                name=f"ewma_{fast}_{slow}_always_on",
                fast=fast,
                slow=slow,
                gate="always_on",
            )
        )
        specs.append(
            StrategySpec(
                name=f"ewma_{fast}_{slow}_er_p60",
                fast=fast,
                slow=slow,
                gate="er",
                er_quantile=0.60,
            )
        )
        specs.append(
            StrategySpec(
                name=f"ewma_{fast}_{slow}_er_p70",
                fast=fast,
                slow=slow,
                gate="er",
                er_quantile=0.70,
            )
        )
        specs.append(
            StrategySpec(
                name=f"ewma_{fast}_{slow}_er_p60_vol_cap2",
                fast=fast,
                slow=slow,
                gate="er_vol",
                er_quantile=0.60,
                vol_floor=0.0,
                vol_ceiling=2.0,
            )
        )
        specs.append(
            StrategySpec(
                name=f"ewma_{fast}_{slow}_er_p60_vol_075_2",
                fast=fast,
                slow=slow,
                gate="er_vol",
                er_quantile=0.60,
                vol_floor=0.75,
                vol_ceiling=2.0,
            )
        )
        specs.append(
            StrategySpec(
                name=f"ewma_{fast}_{slow}_hyst_p70_p40",
                fast=fast,
                slow=slow,
                gate="er_hyst",
                er_quantile=0.70,
                er_exit_quantile=0.40,
            )
        )
        specs.append(
            StrategySpec(
                name=f"ewma_{fast}_{slow}_hyst_p70_p40_vol_cap2",
                fast=fast,
                slow=slow,
                gate="er_hyst_vol",
                er_quantile=0.70,
                er_exit_quantile=0.40,
                vol_floor=0.0,
                vol_ceiling=2.0,
            )
        )
        specs.append(
            StrategySpec(
                name=f"ewma_{fast}_{slow}_hmm3_weekly_p55",
                fast=fast,
                slow=slow,
                gate="hmm3_weekly",
                hmm_prob_entry=0.55,
                hmm_align_with_signal=True,
            )
        )
        specs.append(
            StrategySpec(
                name=f"ewma_{fast}_{slow}_hmm3_sticky_p60_p45",
                fast=fast,
                slow=slow,
                gate="hmm3_weekly_sticky",
                hmm_prob_entry=0.60,
                hmm_prob_exit=0.45,
                hmm_align_with_signal=True,
            )
        )
    return specs


def plot_results(
    bars: pd.DataFrame,
    results: dict[str, pd.DataFrame],
    metrics_rows: list[dict[str, object]],
    out_dir: Path,
) -> None:
    ranked = sorted(metrics_rows, key=lambda r: float(r["sharpe"]), reverse=True)
    names_to_plot: list[str] = []
    for r in ranked:
        name = str(r["name"])
        base = f"ewma_{int(r['fast'])}_{int(r['slow'])}_always_on"
        if name not in names_to_plot:
            names_to_plot.append(name)
        if base in results and base not in names_to_plot:
            names_to_plot.append(base)
        if len(names_to_plot) >= 4:
            break

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    ax0, ax1, ax2 = axes

    btc_nav = bars["open"] / bars["open"].iloc[0]
    ax0.plot(btc_nav.index, btc_nav, color="black", lw=1.2, label="BTC buy & hold")
    ax0.set_yscale("log")
    ax0.set_ylabel("BTC nav (log)")
    ax0.legend(loc="upper left")
    ax0.grid(True, alpha=0.25)

    for name in names_to_plot:
        ax1.plot(results[name].index, results[name]["equity"], lw=1.1, label=name)
    ax1.set_yscale("log")
    ax1.set_ylabel("strategy nav (log)")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(True, alpha=0.25)

    best = ranked[0]["name"]
    best_result = results[str(best)]
    ax2.fill_between(
        best_result.index,
        0.0,
        best_result["regime_on"],
        step="mid",
        alpha=0.35,
        label=f"{best} regime_on",
    )
    ax2.plot(best_result.index, best_result["position"], lw=0.8, label="position")
    ax2.set_ylim(-1.1, 1.1)
    ax2.set_ylabel("gate / position")
    ax2.legend(loc="upper left", fontsize=8)
    ax2.grid(True, alpha=0.25)

    fig.suptitle("BTC-USDC regime-gated EWMA crossover experiment")
    fig.tight_layout()
    fig.savefig(out_dir / "btc_regime_switching_ma.png", dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bars = load_bars(args.lake, args.symbol, args.start_date)
    specs = default_specs()
    results: dict[str, pd.DataFrame] = {}
    rows: list[dict[str, object]] = []

    for spec in specs:
        result = run_strategy(bars, spec, args.cost_bps)
        results[spec.name] = result
        row = metrics(result)
        row.update(
            {
                "name": spec.name,
                "symbol": args.symbol,
                "fast": spec.fast,
                "slow": spec.slow,
                "gate": spec.gate,
                "er_quantile": spec.er_quantile,
                "er_exit_quantile": spec.er_exit_quantile,
                "vol_floor": spec.vol_floor,
                "vol_ceiling": spec.vol_ceiling if math.isfinite(spec.vol_ceiling) else "",
                "hmm_span": spec.hmm_span,
                "hmm_train_days": spec.hmm_train_days,
                "hmm_min_train_days": spec.hmm_min_train_days,
                "hmm_update_days": spec.hmm_update_days,
                "hmm_prob_entry": spec.hmm_prob_entry,
                "hmm_prob_exit": spec.hmm_prob_exit,
                "hmm_align_with_signal": spec.hmm_align_with_signal,
                "cost_bps": args.cost_bps,
            }
        )
        rows.append(row)
        result.to_csv(out_dir / f"{spec.name}.csv")

    metrics_df = pd.DataFrame(rows).sort_values("sharpe", ascending=False)
    metrics_df.to_csv(out_dir / "metrics.csv", index=False)
    (out_dir / "summary.json").write_text(
        json.dumps(metrics_df.to_dict(orient="records"), indent=2, default=str)
    )
    plot_results(bars, results, rows, out_dir)

    print(f"[btc-regime-ma] symbol={args.symbol} rows={len(bars):,}")
    print(f"[btc-regime-ma] wrote {out_dir / 'metrics.csv'}")
    print(metrics_df[
        [
            "name",
            "sharpe",
            "cagr",
            "max_dd",
            "calmar",
            "avg_exposure",
            "pct_regime_on",
            "trade_count",
        ]
    ].to_string(index=False))


if __name__ == "__main__":
    main()
