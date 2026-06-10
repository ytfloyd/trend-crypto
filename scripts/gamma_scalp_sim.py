"""Gamma-scalp simulator — what an underpriced-gamma signal is worth in cash.

Given a (spot, strike, IV, RV, days-to-expiry, contracts) setup, this runs:

  1. Black-Scholes ATM straddle pricing + greeks at inception (r = q = 0).
  2. A seeded single-path GBM simulation of the underlying at the realized-vol
     assumption, with a banded delta-hedge rebalancing policy that logs every
     hedge trade so you can see the gamma scalp happening.
  3. An N-path Monte Carlo ensemble for mean / P10 / P50 / P90 / win rate.
  4. A realized-vol sensitivity table: same trade, scanned across RV grid.

Run without args to reproduce the JPM screener signal (IV 23.8 / RV 30.2 /
315 strike / 30 DTE / 10 straddles). Override any parameter via CLI flags.

Example:
    python scripts/gamma_scalp_sim.py \\
        --spot 313.97 --strike 315 --iv 23.8 --rv 30.2 \\
        --days 30 --contracts 10 --band 25 --paths 500

Deliberately NOT modeled (real P&L will be ~20-40% lower than sim mean):
commissions, bid-ask slippage, IV drift / vega P&L, borrow cost, overnight
gaps, dividends, earnings. This is a clean view of the gamma edge in
isolation.
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np


# ── Black-Scholes (r = q = 0) ────────────────────────────────────────
def _norm_cdf(x: float | np.ndarray) -> float | np.ndarray:
    return 0.5 * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


@dataclass
class StraddleGreeks:
    price: float      # straddle mid, per share
    delta: float      # 2N(d1) - 1
    gamma: float      # 2 * single-side gamma
    theta: float      # per year, negative
    vega: float       # per 1.00 vol change

    @classmethod
    def compute(cls, S: float, K: float, sigma: float, T: float) -> "StraddleGreeks":
        if T <= 0 or sigma <= 0:
            return cls(
                price=max(S - K, 0.0) + max(K - S, 0.0),
                delta=1.0 if S > K else (-1.0 if S < K else 0.0),
                gamma=0.0, theta=0.0, vega=0.0,
            )
        srt = sigma * math.sqrt(T)
        d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / srt
        d2 = d1 - srt
        Nd1 = 0.5 * (1.0 + math.erf(d1 / math.sqrt(2.0)))
        Nd2 = 0.5 * (1.0 + math.erf(d2 / math.sqrt(2.0)))
        call = S * Nd1 - K * Nd2
        put = K * (1.0 - Nd2) - S * (1.0 - Nd1)
        phi = _norm_pdf(d1)
        gamma_single = phi / (S * srt)
        vega_single = S * phi * math.sqrt(T)
        theta_single = -S * phi * sigma / (2.0 * math.sqrt(T))
        return cls(
            price=call + put,
            delta=2.0 * Nd1 - 1.0,
            gamma=2.0 * gamma_single,
            theta=2.0 * theta_single,
            vega=2.0 * vega_single,
        )


# ── Simulation ───────────────────────────────────────────────────────
@dataclass
class SimResult:
    price_path: np.ndarray
    trades: list[dict]           # each: {day, price, side, shares, pos_after}
    initial_premium: float
    terminal_straddle: float
    terminal_hedge_cash_mtm: float
    net_pnl: float
    n_rebalances: int
    max_abs_delta: float


def simulate_one_path(
    S0: float, K: float, iv: float, rv: float,
    days: int, contracts: int, band_shares: int,
    moc_flatten: bool, seed: int,
) -> SimResult:
    rng = np.random.default_rng(seed)
    dt = 1.0 / 252.0
    n_steps = max(1, round((days / 365.0) * 252.0))
    T0 = n_steps * dt

    g0 = StraddleGreeks.compute(S0, K, iv, T0)
    initial_premium = g0.price * 100.0 * contracts

    z = rng.standard_normal(n_steps)
    log_returns = (-0.5 * rv * rv * dt) + (rv * math.sqrt(dt)) * z
    prices = S0 * np.exp(np.cumsum(log_returns))
    price_path = np.concatenate([[S0], prices])

    stock_pos = 0      # signed shares
    cash_from_hedges = 0.0
    max_abs_delta = 0.0
    trades: list[dict] = []

    for step in range(1, n_steps + 1):
        price = float(price_path[step])
        t_remaining = (n_steps - step) * dt
        g = StraddleGreeks.compute(price, K, iv, t_remaining)
        straddle_delta_shares = g.delta * 100.0 * contracts
        net_delta = straddle_delta_shares + stock_pos
        max_abs_delta = max(max_abs_delta, abs(net_delta))

        is_last = step == n_steps
        exceeds_band = abs(net_delta) > band_shares * contracts
        if exceeds_band or (moc_flatten and is_last):
            shares_to_trade = -net_delta
            if abs(shares_to_trade) >= 1:
                cash_from_hedges += -shares_to_trade * price
                stock_pos += shares_to_trade
                trades.append({
                    "day": step,
                    "price": price,
                    "side": "buy" if shares_to_trade > 0 else "sell",
                    "shares": abs(round(shares_to_trade)),
                    "pos_after": round(stock_pos),
                })

    terminal_price = float(price_path[-1])
    terminal_straddle = (
        max(terminal_price - K, 0.0) + max(K - terminal_price, 0.0)
    ) * 100.0 * contracts
    terminal_hedge_cash_mtm = stock_pos * terminal_price + cash_from_hedges
    net_pnl = -initial_premium + terminal_straddle + terminal_hedge_cash_mtm

    return SimResult(
        price_path=price_path,
        trades=trades,
        initial_premium=initial_premium,
        terminal_straddle=terminal_straddle,
        terminal_hedge_cash_mtm=terminal_hedge_cash_mtm,
        net_pnl=net_pnl,
        n_rebalances=len(trades),
        max_abs_delta=max_abs_delta,
    )


def ensemble(
    S0: float, K: float, iv: float, rv: float,
    days: int, contracts: int, band_shares: int,
    moc_flatten: bool, seed: int, n_paths: int,
) -> dict:
    pnls = np.empty(n_paths)
    rebals = np.empty(n_paths, dtype=int)
    for i in range(n_paths):
        r = simulate_one_path(
            S0, K, iv, rv, days, contracts, band_shares, moc_flatten, seed + i,
        )
        pnls[i] = r.net_pnl
        rebals[i] = r.n_rebalances
    return {
        "mean": float(pnls.mean()),
        "std": float(pnls.std()),
        "p10": float(np.percentile(pnls, 10)),
        "p50": float(np.percentile(pnls, 50)),
        "p90": float(np.percentile(pnls, 90)),
        "win_rate": float((pnls > 0).mean()),
        "avg_rebalances": float(rebals.mean()),
    }


# ── Pretty-print helpers ─────────────────────────────────────────────
def fmt_cash(x: float) -> str:
    sign = "-" if x < 0 else "+"
    return f"{sign}${abs(x):,.0f}"


def fmt_pct(x: float) -> str:
    return f"{x * 100:,.1f}%"


def section(title: str) -> None:
    line = "─" * max(70, len(title) + 4)
    print()
    print(line)
    print(f"  {title}")
    print(line)


def print_trades(trades: list[dict], limit: int = 12) -> None:
    if not trades:
        print("  (no rebalances — path never crossed the band)")
        return
    print(f"  {'Day':>5}  {'Price':>10}  {'Side':<5}  {'Shares':>7}  {'Pos after':>10}")
    for tr in trades[:limit]:
        print(
            f"  {tr['day']:>5}  ${tr['price']:>9.2f}  "
            f"{tr['side']:<5}  {tr['shares']:>7}  {tr['pos_after']:>10}"
        )
    if len(trades) > limit:
        print(f"  … ({len(trades) - limit} more)")


# ── CLI ──────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Gamma-scalp simulator for an underpriced-gamma signal.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--spot", type=float, default=313.97, help="Underlying price")
    p.add_argument("--strike", type=float, default=315.0, help="ATM-ish strike")
    p.add_argument("--iv", type=float, default=23.8, help="Implied vol (percent)")
    p.add_argument("--rv", type=float, default=30.2, help="Realized vol assumption (percent)")
    p.add_argument("--days", type=int, default=30, help="Calendar days to expiry")
    p.add_argument("--contracts", type=int, default=10, help="Number of straddles")
    p.add_argument("--band", type=int, default=25, help="Rebalance band (shares / contract)")
    p.add_argument("--no-moc", action="store_true", help="Disable MOC flatten at expiry")
    p.add_argument("--seed", type=int, default=42, help="Seed for single-path realization")
    p.add_argument("--paths", type=int, default=500, help="Monte-Carlo path count for ensemble")
    p.add_argument(
        "--rv-grid",
        type=str,
        default="15,20,25,30,35,40",
        help="Comma-separated realized-vol grid for sensitivity table",
    )
    p.add_argument("--symbol", type=str, default="JPM", help="Ticker label (cosmetic only)")
    return p.parse_args()


def main() -> None:
    a = parse_args()
    iv = a.iv / 100.0
    rv = a.rv / 100.0
    moc = not a.no_moc

    section(f"{a.symbol} gamma-scalp simulator — trade setup")
    print(f"  Spot:              ${a.spot:,.2f}")
    print(f"  Strike:            ${a.strike:,.2f}")
    print(f"  Implied vol:       {fmt_pct(iv)}")
    print(f"  Realized vol:      {fmt_pct(rv)}  (assumption)")
    print(f"  Days to expiry:    {a.days}")
    print(f"  Straddles:         {a.contracts}")
    print(f"  Rebalance band:    {a.band} shares / contract")
    print(f"  MOC flatten:       {moc}")
    print(f"  Path seed:         {a.seed}")
    print(f"  Ensemble paths:    {a.paths}")

    # Greeks at inception
    T0 = a.days / 365.0
    g0 = StraddleGreeks.compute(a.spot, a.strike, iv, T0)

    section("Position at inception")
    premium = g0.price * 100 * a.contracts
    daily_theta_dollars = -(g0.theta / 365.0) * 100 * a.contracts
    print(f"  Premium paid:      ${premium:,.0f}  ({a.contracts} straddle(s) × ${g0.price:.2f} × 100)")
    print(f"  Delta / straddle:  {g0.delta:+.4f}")
    print(f"  Gamma / straddle:  {g0.gamma * 100:.4f}   (per $1, ×100 shares)")
    print(f"  Theta / day:       -${daily_theta_dollars:,.0f}   (bleed if nothing moves)")
    print(f"  Vega / vol pt:     ${g0.vega / 100 * 100 * a.contracts:,.0f}")

    # Single path
    section(f"One-path realization · seed {a.seed}")
    r = simulate_one_path(
        a.spot, a.strike, iv, rv, a.days, a.contracts, a.band, moc, a.seed,
    )
    print(f"  Starting price:    ${r.price_path[0]:.2f}")
    print(f"  Ending price:      ${r.price_path[-1]:.2f}")
    print(f"  Path range:        ${r.price_path.min():.2f} – ${r.price_path.max():.2f}")
    print(f"  Max net delta:     {int(r.max_abs_delta)} shares")
    print(f"  Rebalances:        {r.n_rebalances}")
    print()
    print(f"  Option-leg P&L:    {fmt_cash(r.terminal_straddle - r.initial_premium)}")
    print(f"  Hedge-leg P&L:     {fmt_cash(r.terminal_hedge_cash_mtm)}")
    print(f"  NET P&L:           {fmt_cash(r.net_pnl)}")

    section("First hedge trades (see the gamma scalp)")
    print_trades(r.trades, limit=12)

    # Ensemble
    section(f"Expected P&L over {a.paths} random paths (same trade, different luck)")
    e = ensemble(
        a.spot, a.strike, iv, rv, a.days, a.contracts, a.band, moc, a.seed, a.paths,
    )
    print(f"  Mean:              {fmt_cash(e['mean'])}")
    print(f"  Median (P50):      {fmt_cash(e['p50'])}")
    print(f"  P10 / P90:         {fmt_cash(e['p10'])}  /  {fmt_cash(e['p90'])}")
    print(f"  Std dev:           ${e['std']:,.0f}")
    print(f"  Win rate:          {fmt_pct(e['win_rate'])}")
    print(f"  Avg rebalances:    {e['avg_rebalances']:.1f}")

    # Sensitivity
    section("Sensitivity to realized vol")
    grid = [float(x.strip()) / 100.0 for x in a.rv_grid.split(",") if x.strip()]
    paths_per_cell = max(100, a.paths // 4)
    header = f"  {'RV':>8}  {'P10':>12}  {'Median':>12}  {'P90':>12}  {'Win %':>8}  {'Mean':>12}"
    print(header)
    print("  " + "─" * (len(header) - 2))
    for rv_try in grid:
        es = ensemble(
            a.spot, a.strike, iv, rv_try, a.days, a.contracts, a.band, moc,
            a.seed, paths_per_cell,
        )
        print(
            f"  {fmt_pct(rv_try):>8}  "
            f"{fmt_cash(es['p10']):>12}  "
            f"{fmt_cash(es['p50']):>12}  "
            f"{fmt_cash(es['p90']):>12}  "
            f"{fmt_pct(es['win_rate']):>8}  "
            f"{fmt_cash(es['mean']):>12}"
        )

    section("Notes")
    print("  Premium is fixed cash outflow up front.")
    print("  Option-leg P&L = terminal |S_T − K| × 100 × contracts − premium.")
    print("  Hedge-leg P&L = cumulative 'sell-high / buy-low' that the band policy forces.")
    print("  Real-world haircut: subtract ~20–40% of the mean for commissions, slippage,")
    print("  and vega drag from IV re-pricing. Borrow cost on short JPM stock ≈ 0.")
    print()


if __name__ == "__main__":
    main()
