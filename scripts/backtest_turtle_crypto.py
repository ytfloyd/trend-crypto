#!/usr/bin/env python3
"""
Turtle Crypto Trading System — Daily Backtest
==============================================
Implements the long-only, spot-crypto Turtle adaptation described in the
project spec:

- Universe: Coinbase USD pairs, daily bars, stablecoin/wrapped-asset exclusion
- Entries:  S1 (20d) and S2 (55d) Donchian breakouts on daily close
            (confirmed at close, executed at next open)
- Sizing:   1 unit = (equity x risk) / (2 x ATR20$), fixed account equity
- Exits:    (a) 2N hard stop (intraday, checked vs. daily low)
            (b) 10d Donchian low close (next-bar open)
- Portfolio: max 20% heat, max 20% notional/order, 30 orders/day cap
- Ranking:  composite breakout_strength * trend_confirm * liquidity, heat-cap pruned

Run:

    python scripts/backtest_turtle_crypto.py \
        --db /Users/russellfloyd/Dropbox/NRT/nrt_dev/data/market.duckdb \
        --start 2018-01-01 \
        --end   2026-03-01 \
        --out   artifacts/backtests/turtle_crypto

Outputs written to ``--out``:

- ``trades.csv``        one row per closed position
- ``daily_log.csv``     daily portfolio value, heat, cash, positions count
- ``summary.json``      headline performance metrics
- ``equity.png``        equity curve vs. BTC buy-and-hold (if matplotlib)

The script is self-contained: it reads ``bars_1d_clean`` directly and does
not depend on ``backtest.portfolio_engine`` (which assumes target-weight
strategies).  It is long-only, non-compounding, with next-bar-open entry
execution and intraday stop-loss handling.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("turtle_crypto")


# ═══════════════════════════════════════════════════════════════════════════════
# 1. CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

# Stablecoins / wrapped / fiat-pegged bases to exclude (spec §1).
EXCLUDED_BASES: set[str] = {
    "USDT", "USDC", "DAI", "BUSD", "TUSD", "USDP", "GUSD", "FRAX", "PYUSD",
    "FDUSD", "EURC", "EURT", "GBPT", "GYEN", "USDS", "UST", "MIM", "LUSD",
    "SUSD", "CRVUSD", "GHO", "MKUSD", "WBTC", "CBBTC", "CBETH", "WETH",
    "STETH", "RETH", "MSOL", "PAX", "HUSD", "TRIBE", "FEI", "ALUSD", "RAI",
}


@dataclass(frozen=True)
class TurtleConfig:
    # Channel lengths
    s1_entry_n: int = 20
    s1_exit_n: int = 10
    s2_entry_n: int = 55
    s2_exit_n: int = 20
    atr_n: int = 20
    ret_n: int = 55

    # Classification thresholds
    s2_extended_pct: float = 140.0   # S2 channel % > 140 → half unit
    s1_weak_ret55: float = 0.05      # 55d return <= 5% → half unit

    # Sizing / risk
    starting_capital: float = 20_000.0
    risk_per_unit: float = 0.005     # 0.5% of (fixed) account equity
    atr_stop_mult: float = 2.0

    # Universe filters (applied on a rolling basis each day)
    min_scanner_vol_usd: float = 50_000.0
    min_exec_vol_usd: float = 100_000.0
    max_atr_pct: float = 12.0        # ATR% of close
    min_history_bars: int = 56

    # Portfolio constraints
    max_heat: float = 0.20           # 20% of account
    max_notional_pct: float = 0.20   # 20% per order
    max_daily_orders: int = 30

    # Execution assumptions
    slippage_bps: float = 50.0       # 0.5%
    commission_bps: float = 60.0     # 0.6% taker

    # Stop-loss behaviour
    stop_limit_slippage: float = 0.005  # 0.5% from stop (spec §6a)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def _strip_base(symbol: str) -> str:
    # "BTC-USD" → "BTC"
    return symbol.split("-", 1)[0].upper()


def load_daily_panel(
    db_path: str,
    start: str | None,
    end: str | None,
) -> pd.DataFrame:
    """Load Coinbase USD daily bars from DuckDB and apply static filters.

    Returns a long-format DataFrame indexed by (ts, symbol) with columns
    open/high/low/close/volume, with ts normalised to tz-naive UTC-day.
    """
    conn = duckdb.connect(db_path, read_only=True)

    bounds = []
    if start:
        bounds.append(f"ts >= TIMESTAMP '{start}'")
    if end:
        bounds.append(f"ts <= TIMESTAMP '{end}'")
    where_dates = (" AND " + " AND ".join(bounds)) if bounds else ""

    sql = f"""
        SELECT
            symbol,
            CAST(ts AS TIMESTAMP)                 AS ts,
            open, high, low, close, volume,
            close * volume                        AS dollar_volume
        FROM bars_1d_clean
        WHERE symbol LIKE '%-USD'
        {where_dates}
        ORDER BY symbol, ts
    """
    df = conn.execute(sql).fetchdf()
    conn.close()

    if df.empty:
        raise RuntimeError(
            f"No rows returned from {db_path}::bars_1d_clean (check date range)"
        )

    # Exclude stablecoin / wrapped bases.
    df["base"] = df["symbol"].map(_strip_base)
    df = df[~df["base"].isin(EXCLUDED_BASES)].drop(columns=["base"]).copy()

    # Normalise to calendar-day timestamps so multiple rows per day cannot
    # exist (defensive: bars_1d_clean is already daily).
    df["ts"] = pd.to_datetime(df["ts"]).dt.tz_localize(None).dt.normalize()
    df = (
        df.sort_values(["symbol", "ts"])
          .drop_duplicates(subset=["symbol", "ts"], keep="last")
          .reset_index(drop=True)
    )
    return df


def to_panel(df: pd.DataFrame, field_name: str) -> pd.DataFrame:
    """Pivot long frame to date x symbol panel of one OHLCV column."""
    panel = df.pivot(index="ts", columns="symbol", values=field_name)
    panel = panel.sort_index()
    return panel


# ═══════════════════════════════════════════════════════════════════════════════
# 3. INDICATORS (vectorised, no look-ahead)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_indicators(
    opens: pd.DataFrame,
    highs: pd.DataFrame,
    lows: pd.DataFrame,
    closes: pd.DataFrame,
    volumes: pd.DataFrame,
    cfg: TurtleConfig,
) -> dict[str, pd.DataFrame]:
    """Compute all Turtle indicators in vectorised form.

    All channel/ATR calculations exclude the current bar (shift(1)) to
    avoid look-ahead per spec §2.
    """
    # --- True Range & ATR ------------------------------------------------------
    prev_close = closes.shift(1)
    hl = (highs - lows).to_numpy()
    hc = (highs - prev_close).abs().to_numpy()
    lc = (lows - prev_close).abs().to_numpy()
    tr_np = np.maximum(np.maximum(hl, hc), lc)
    tr = pd.DataFrame(tr_np, index=highs.index, columns=highs.columns)
    atr = tr.rolling(cfg.atr_n, min_periods=cfg.atr_n).mean()
    atr_pct = (atr / closes) * 100.0

    # --- Donchian channels (PRIOR N bars — shift(1)) ---------------------------
    s1_entry_hi = highs.shift(1).rolling(cfg.s1_entry_n, min_periods=cfg.s1_entry_n).max()
    s1_exit_lo = lows.shift(1).rolling(cfg.s1_exit_n, min_periods=cfg.s1_exit_n).min()
    s2_entry_hi = highs.shift(1).rolling(cfg.s2_entry_n, min_periods=cfg.s2_entry_n).max()
    s2_exit_lo = lows.shift(1).rolling(cfg.s2_exit_n, min_periods=cfg.s2_exit_n).min()

    # --- Channel position % ----------------------------------------------------
    s1_range = (s1_entry_hi - s1_exit_lo).replace(0.0, np.nan)
    s1_pct = (closes - s1_exit_lo) / s1_range * 100.0
    s2_range = (s2_entry_hi - s2_exit_lo).replace(0.0, np.nan)
    s2_pct = (closes - s2_exit_lo) / s2_range * 100.0

    # --- Breakout flags --------------------------------------------------------
    s1_long = closes >= s1_entry_hi
    s2_long = closes >= s2_entry_hi
    s1_exit = closes <= s1_exit_lo

    # --- Trend filter ----------------------------------------------------------
    ret_55 = closes / closes.shift(cfg.ret_n) - 1.0

    # --- Liquidity (24h USD volume proxy = close * daily volume) --------------
    dollar_vol = closes * volumes
    dollar_vol_rolling = dollar_vol.rolling(5, min_periods=1).mean()

    # --- History sufficiency ---------------------------------------------------
    has_history = closes.notna().rolling(cfg.min_history_bars, min_periods=cfg.min_history_bars).sum() \
                        >= cfg.min_history_bars

    return {
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
        "atr": atr,
        "atr_pct": atr_pct,
        "s1_entry_hi": s1_entry_hi,
        "s1_exit_lo": s1_exit_lo,
        "s2_entry_hi": s2_entry_hi,
        "s2_exit_lo": s2_exit_lo,
        "s1_pct": s1_pct,
        "s2_pct": s2_pct,
        "s1_long": s1_long,
        "s2_long": s2_long,
        "s1_exit": s1_exit,
        "ret_55": ret_55,
        "dollar_vol": dollar_vol,
        "dollar_vol_rolling": dollar_vol_rolling,
        "has_history": has_history,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 4. SIMULATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Position:
    symbol: str
    qty: float
    entry_ts: pd.Timestamp
    entry_price: float            # actual fill including slippage
    entry_close: float            # close that triggered signal
    atr: float                    # ATR at entry (for risk / stop)
    stop_price: float
    unit_scale: float             # 1.0 full unit, 0.5 half unit
    label: str                    # S1 LONG / S2 LONG / S1 WEAK / S2 EXTENDED
    risk_dollars: float           # initial risk committed

    @property
    def notional(self) -> float:
        return self.qty * self.entry_price


@dataclass
class Trade:
    symbol: str
    label: str
    unit_scale: float
    entry_ts: pd.Timestamp
    entry_price: float
    qty: float
    exit_ts: pd.Timestamp
    exit_price: float
    exit_reason: str               # 'stop' | 'donchian' | 'eof'
    pnl: float
    fees: float
    hold_days: int


def _breakout_strength(
    label: str, close: float, s1_hi: float, s2_hi: float, atr: float, s1_pct: float,
) -> float:
    if label.startswith("S2"):
        return 2.0 + (close - s2_hi) / max(atr, 1e-9)
    if label.startswith("S1"):
        return 1.0 + (close - s1_hi) / max(atr, 1e-9)
    return (s1_pct if not math.isnan(s1_pct) else 0.0) / 100.0


def _classify_signal(
    s1_long: bool,
    s2_long: bool,
    s2_pct: float,
    ret_55: float,
    s1_pct: float,
    cfg: TurtleConfig,
) -> str | None:
    """Return S2 LONG / S2 EXTENDED / S1 LONG / S1 WEAK / None. Spec §3."""
    if s2_long:
        if math.isnan(s2_pct):
            return "S2 LONG"
        return "S2 LONG" if s2_pct <= cfg.s2_extended_pct else "S2 EXTENDED"
    if s1_long:
        r = 0.0 if math.isnan(ret_55) else ret_55
        return "S1 LONG" if r > cfg.s1_weak_ret55 else "S1 WEAK"
    return None


def run_backtest(
    ind: dict[str, pd.DataFrame],
    cfg: TurtleConfig,
) -> tuple[list[Trade], pd.DataFrame]:
    """Day-by-day portfolio simulation. Returns (trades, daily_log)."""

    dates = ind["close"].index
    symbols = list(ind["close"].columns)

    # Convert panels to numpy for speed (N days x M symbols).
    arrs = {k: v.to_numpy() for k, v in ind.items() if v.ndim == 2}
    sym_idx = {s: i for i, s in enumerate(symbols)}

    positions: dict[str, Position] = {}
    trades: list[Trade] = []

    # Orders queued from the close of day t, executed at open of day t+1.
    pending_sells: list[str] = []
    pending_buys: list[dict[str, Any]] = []

    cash = cfg.starting_capital
    account_equity = cfg.starting_capital  # fixed for sizing (spec §4)

    log_rows: list[dict[str, Any]] = []

    for t in range(len(dates)):
        ts = dates[t]
        o = arrs["open"][t]
        h = arrs["high"][t]
        lo = arrs["low"][t]
        c = arrs["close"][t]

        # ─────────────────────────────────────────────────
        # 1. EXECUTE PENDING ORDERS AT TODAY'S OPEN
        # ─────────────────────────────────────────────────

        # 1a. Sells first (frees capital for new buys).
        for sym in pending_sells:
            if sym not in positions:
                continue
            i = sym_idx[sym]
            open_px = o[i]
            if not np.isfinite(open_px) or open_px <= 0:
                # No trade bar — carry forward, try again next day.
                continue
            pos = positions[sym]
            fill = open_px * (1.0 - cfg.slippage_bps / 10_000.0)
            gross = pos.qty * fill
            fees = gross * (cfg.commission_bps / 10_000.0)
            cash += gross - fees
            pnl = (fill - pos.entry_price) * pos.qty - fees
            trades.append(Trade(
                symbol=sym, label=pos.label, unit_scale=pos.unit_scale,
                entry_ts=pos.entry_ts, entry_price=pos.entry_price, qty=pos.qty,
                exit_ts=ts, exit_price=fill, exit_reason="donchian",
                pnl=pnl, fees=fees,
                hold_days=(ts - pos.entry_ts).days,
            ))
            del positions[sym]
        pending_sells = []

        # 1b. Buys.  Each pending_buy already carries its sized quantity and
        # ATR-at-signal; we just apply slippage + commission and commit cash.
        for order in pending_buys:
            sym = order["symbol"]
            if sym in positions:
                continue  # max 1 unit per asset (spec §3)
            i = sym_idx[sym]
            open_px = o[i]
            if not np.isfinite(open_px) or open_px <= 0:
                continue
            fill = open_px * (1.0 + cfg.slippage_bps / 10_000.0)
            qty = order["qty"]
            gross = qty * fill
            fees = gross * (cfg.commission_bps / 10_000.0)
            # Guard against negative cash from big gap opens: shrink qty to fit.
            max_afford = max(cash - fees, 0.0) / max(fill, 1e-9)
            if qty > max_afford:
                qty = max_afford
                gross = qty * fill
                fees = gross * (cfg.commission_bps / 10_000.0)
            if qty <= 0 or gross < 1.0:
                continue
            cash -= gross + fees
            atr_at_entry = order["atr"]
            stop_price = order["entry_close"] - cfg.atr_stop_mult * atr_at_entry
            positions[sym] = Position(
                symbol=sym, qty=qty, entry_ts=ts, entry_price=fill,
                entry_close=order["entry_close"], atr=atr_at_entry,
                stop_price=stop_price, unit_scale=order["unit_scale"],
                label=order["label"],
                risk_dollars=order["risk_dollars"],
            )
        pending_buys = []

        # ─────────────────────────────────────────────────
        # 2. INTRADAY STOP-LOSS CHECK (spec §6a, priority over Donchian)
        # ─────────────────────────────────────────────────
        stopped_out: list[str] = []
        for sym, pos in positions.items():
            i = sym_idx[sym]
            lo_px = lo[i]
            op_px = o[i]
            if not np.isfinite(lo_px):
                continue
            if lo_px <= pos.stop_price:
                # Fill at stop price (or gap-down open if below stop).
                fill = min(pos.stop_price, op_px) if np.isfinite(op_px) else pos.stop_price
                # Apply limit-style slippage tolerance (0.5% below stop).
                fill = fill * (1.0 - cfg.stop_limit_slippage)
                gross = pos.qty * fill
                fees = gross * (cfg.commission_bps / 10_000.0)
                cash += gross - fees
                pnl = (fill - pos.entry_price) * pos.qty - fees
                trades.append(Trade(
                    symbol=sym, label=pos.label, unit_scale=pos.unit_scale,
                    entry_ts=pos.entry_ts, entry_price=pos.entry_price, qty=pos.qty,
                    exit_ts=ts, exit_price=fill, exit_reason="stop",
                    pnl=pnl, fees=fees,
                    hold_days=(ts - pos.entry_ts).days,
                ))
                stopped_out.append(sym)
        for s in stopped_out:
            del positions[s]

        # ─────────────────────────────────────────────────
        # 3. AT CLOSE: EXIT SIGNALS (queue for next open)
        # ─────────────────────────────────────────────────
        for sym, pos in list(positions.items()):
            i = sym_idx[sym]
            close_px = c[i]
            exit_lo = arrs["s1_exit_lo"][t][i]
            if np.isfinite(close_px) and np.isfinite(exit_lo) and close_px <= exit_lo:
                pending_sells.append(sym)

        # ─────────────────────────────────────────────────
        # 4. AT CLOSE: ENTRY SIGNALS + RANKING + HEAT CAP
        # ─────────────────────────────────────────────────
        candidates: list[dict[str, Any]] = []
        for j, sym in enumerate(symbols):
            if sym in positions or sym in pending_sells:
                continue
            close_px = c[j]
            if not np.isfinite(close_px) or close_px <= 0:
                continue

            if not bool(arrs["has_history"][t][j]):
                continue

            atr_val = arrs["atr"][t][j]
            atr_pct_val = arrs["atr_pct"][t][j]
            dvol = arrs["dollar_vol_rolling"][t][j]
            if not np.isfinite(atr_val) or atr_val <= 0:
                continue
            if not np.isfinite(dvol) or dvol < cfg.min_scanner_vol_usd:
                continue
            # Execution filter (spec §1).
            if dvol < cfg.min_exec_vol_usd:
                continue
            if not np.isfinite(atr_pct_val) or atr_pct_val > cfg.max_atr_pct:
                continue

            s1_long = bool(arrs["s1_long"][t][j])
            s2_long = bool(arrs["s2_long"][t][j])
            if not (s1_long or s2_long):
                continue
            s2_pct = arrs["s2_pct"][t][j]
            s1_pct = arrs["s1_pct"][t][j]
            ret55 = arrs["ret_55"][t][j]
            label = _classify_signal(s1_long, s2_long, s2_pct, ret55, s1_pct, cfg)
            if label is None:
                continue

            # Sizing
            unit_scale = 0.5 if label in ("S2 EXTENDED", "S1 WEAK") else 1.0
            base_size = (account_equity * cfg.risk_per_unit) / (cfg.atr_stop_mult * atr_val)
            qty = base_size * unit_scale

            # Per-order notional cap (spec §7).
            max_notional = account_equity * cfg.max_notional_pct
            notional = qty * close_px
            if notional <= 0:
                continue
            if notional > max_notional:
                qty = max_notional / close_px
                notional = qty * close_px

            # Risk committed (what heat cap measures).
            risk_dollars = qty * cfg.atr_stop_mult * atr_val

            # Ranking score
            strength = _breakout_strength(
                label,
                close=close_px,
                s1_hi=arrs["s1_entry_hi"][t][j],
                s2_hi=arrs["s2_entry_hi"][t][j],
                atr=atr_val,
                s1_pct=s1_pct,
            )
            trend_conf = 1.0 + max(ret55 if np.isfinite(ret55) else 0.0, 0.0)
            liquidity = math.log10(max(dvol, 10.0))
            score = strength * trend_conf * liquidity

            candidates.append({
                "symbol": sym,
                "label": label,
                "unit_scale": unit_scale,
                "entry_close": float(close_px),
                "atr": float(atr_val),
                "qty": float(qty),
                "notional": float(notional),
                "risk_dollars": float(risk_dollars),
                "score": float(score),
            })

        # Sort by score desc; prune by heat cap and daily order cap.
        candidates.sort(key=lambda x: x["score"], reverse=True)

        current_heat = sum(p.risk_dollars for p in positions.values()) / account_equity
        max_heat_dollars = cfg.max_heat * account_equity
        heat_used = current_heat * account_equity

        accepted: list[dict[str, Any]] = []
        for cand in candidates:
            if len(accepted) >= cfg.max_daily_orders:
                break
            if heat_used + cand["risk_dollars"] > max_heat_dollars + 1e-6:
                # Skip; try lower-ranked ones that may fit.
                continue
            accepted.append(cand)
            heat_used += cand["risk_dollars"]

        pending_buys.extend(accepted)

        # ─────────────────────────────────────────────────
        # 5. LOG DAILY STATE
        # ─────────────────────────────────────────────────
        mark_val = 0.0
        for sym, pos in positions.items():
            i = sym_idx[sym]
            px = c[i]
            if np.isfinite(px):
                mark_val += pos.qty * px
            else:
                mark_val += pos.qty * pos.entry_price
        equity = cash + mark_val
        heat_pct = sum(p.risk_dollars for p in positions.values()) / account_equity
        log_rows.append({
            "ts": ts,
            "equity": equity,
            "cash": cash,
            "positions_value": mark_val,
            "n_positions": len(positions),
            "heat_pct": heat_pct,
            "n_candidates": len(candidates),
            "n_new_orders": len(accepted),
        })

    # Force-close any positions still open at the last bar for clean stats.
    last_ts = dates[-1]
    for sym, pos in list(positions.items()):
        i = sym_idx[sym]
        last_close = arrs["close"][-1][i]
        if not np.isfinite(last_close) or last_close <= 0:
            last_close = pos.entry_price
        fill = last_close * (1.0 - cfg.slippage_bps / 10_000.0)
        gross = pos.qty * fill
        fees = gross * (cfg.commission_bps / 10_000.0)
        cash += gross - fees
        pnl = (fill - pos.entry_price) * pos.qty - fees
        trades.append(Trade(
            symbol=sym, label=pos.label, unit_scale=pos.unit_scale,
            entry_ts=pos.entry_ts, entry_price=pos.entry_price, qty=pos.qty,
            exit_ts=last_ts, exit_price=fill, exit_reason="eof",
            pnl=pnl, fees=fees,
            hold_days=(last_ts - pos.entry_ts).days,
        ))

    daily_log = pd.DataFrame(log_rows).set_index("ts")
    return trades, daily_log


# ═══════════════════════════════════════════════════════════════════════════════
# 5. METRICS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_metrics(daily_log: pd.DataFrame, trades: list[Trade], cfg: TurtleConfig) -> dict[str, Any]:
    eq = daily_log["equity"].astype(float)
    rets = eq.pct_change().fillna(0.0)

    years = max(len(eq) / 365.25, 1e-9)
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1.0 / years) - 1.0 if eq.iloc[0] > 0 else 0.0
    vol = rets.std(ddof=0) * math.sqrt(365.25)
    sharpe = (rets.mean() * 365.25) / (rets.std(ddof=0) * math.sqrt(365.25)) \
        if rets.std(ddof=0) > 0 else 0.0
    # Sortino on downside
    neg = rets[rets < 0]
    sortino = (rets.mean() * 365.25) / (neg.std(ddof=0) * math.sqrt(365.25)) \
        if len(neg) > 0 and neg.std(ddof=0) > 0 else 0.0
    peak = eq.cummax()
    dd = eq / peak - 1.0
    mdd = float(dd.min())
    calmar = cagr / abs(mdd) if mdd < 0 else float("nan")

    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl <= 0]
    gross_win = sum(t.pnl for t in wins)
    gross_loss = abs(sum(t.pnl for t in losses))
    profit_factor = gross_win / gross_loss if gross_loss > 0 else float("nan")
    avg_hold = (sum(t.hold_days for t in trades) / len(trades)) if trades else 0.0

    return {
        "start": str(daily_log.index[0].date()),
        "end": str(daily_log.index[-1].date()),
        "starting_capital": cfg.starting_capital,
        "ending_equity": float(eq.iloc[-1]),
        "total_return": float(eq.iloc[-1] / eq.iloc[0] - 1.0),
        "cagr": float(cagr),
        "ann_vol": float(vol),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_drawdown": mdd,
        "calmar": float(calmar) if calmar == calmar else None,
        "num_trades": len(trades),
        "win_rate": (len(wins) / len(trades)) if trades else 0.0,
        "profit_factor": float(profit_factor) if profit_factor == profit_factor else None,
        "avg_hold_days": float(avg_hold),
        "exits_stop": sum(1 for t in trades if t.exit_reason == "stop"),
        "exits_donchian": sum(1 for t in trades if t.exit_reason == "donchian"),
        "exits_eof": sum(1 for t in trades if t.exit_reason == "eof"),
        "avg_positions": float(daily_log["n_positions"].mean()),
        "max_positions": int(daily_log["n_positions"].max()),
        "avg_heat_pct": float(daily_log["heat_pct"].mean()),
        "max_heat_pct": float(daily_log["heat_pct"].max()),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 6. REPORTING
# ═══════════════════════════════════════════════════════════════════════════════

def save_outputs(
    out_dir: Path,
    trades: list[Trade],
    daily_log: pd.DataFrame,
    summary: dict[str, Any],
    closes: pd.DataFrame,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # trades.csv
    trades_df = pd.DataFrame([t.__dict__ for t in trades])
    if not trades_df.empty:
        trades_df = trades_df.sort_values("entry_ts").reset_index(drop=True)
    trades_df.to_csv(out_dir / "trades.csv", index=False)

    # daily_log.csv
    daily_log.to_csv(out_dir / "daily_log.csv")

    # summary.json
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    # Equity chart (best-effort).
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(11, 5))
        eq = daily_log["equity"]
        ax.plot(eq.index, eq.values, label="Turtle Crypto", color="#1f77b4", lw=1.4)
        if "BTC-USD" in closes.columns:
            btc = closes["BTC-USD"].reindex(eq.index).ffill()
            btc_eq = summary["starting_capital"] * (btc / btc.iloc[0])
            ax.plot(btc_eq.index, btc_eq.values, label="BTC Buy & Hold",
                    color="#ff7f0e", lw=1.0, alpha=0.8)
        ax.set_title(
            f"Turtle Crypto — {summary['start']} → {summary['end']}  "
            f"(CAGR {summary['cagr']*100:.1f}%, Sharpe {summary['sharpe']:.2f}, "
            f"MaxDD {summary['max_drawdown']*100:.1f}%)"
        )
        ax.set_ylabel("Account Equity ($)")
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(out_dir / "equity.png", dpi=140)
        plt.close(fig)
    except Exception as exc:  # pragma: no cover - plotting optional
        logger.warning("Skipping equity chart (%s)", exc)


# ═══════════════════════════════════════════════════════════════════════════════
# 7. CLI
# ═══════════════════════════════════════════════════════════════════════════════

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Turtle Crypto daily backtest")
    p.add_argument(
        "--db",
        default=str(PROJECT_ROOT.parent / "data" / "market.duckdb"),
        help="Path to DuckDB with bars_1d_clean table.",
    )
    p.add_argument("--start", default="2018-01-01", help="Backtest start (YYYY-MM-DD).")
    p.add_argument("--end", default=None, help="Backtest end (YYYY-MM-DD).")
    p.add_argument(
        "--out",
        default=str(PROJECT_ROOT / "artifacts" / "backtests" / "turtle_crypto"),
        help="Output directory for artefacts.",
    )
    p.add_argument("--capital", type=float, default=20_000.0, help="Starting capital.")
    p.add_argument("--risk", type=float, default=0.005, help="Risk per unit (fraction).")
    p.add_argument("--max-heat", type=float, default=0.20, help="Max portfolio heat.")
    p.add_argument("--max-atr-pct", type=float, default=12.0, help="Max ATR%% of close.")
    p.add_argument(
        "--min-exec-vol",
        type=float,
        default=100_000.0,
        help="Min 24h USD volume (execution).",
    )
    p.add_argument("--commission-bps", type=float, default=60.0, help="Commission (bps).")
    p.add_argument("--slippage-bps", type=float, default=50.0, help="Slippage (bps).")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)

    cfg = TurtleConfig(
        starting_capital=args.capital,
        risk_per_unit=args.risk,
        max_heat=args.max_heat,
        max_atr_pct=args.max_atr_pct,
        min_exec_vol_usd=args.min_exec_vol,
        commission_bps=args.commission_bps,
        slippage_bps=args.slippage_bps,
    )

    logger.info("Loading panel from %s …", args.db)
    df = load_daily_panel(args.db, args.start, args.end)
    logger.info("Loaded %d rows across %d symbols",
                len(df), df["symbol"].nunique())

    opens = to_panel(df, "open")
    highs = to_panel(df, "high")
    lows = to_panel(df, "low")
    closes = to_panel(df, "close")
    volumes = to_panel(df, "volume")

    logger.info("Computing indicators on %d dates x %d symbols",
                closes.shape[0], closes.shape[1])
    ind = compute_indicators(opens, highs, lows, closes, volumes, cfg)

    logger.info("Running simulation …")
    trades, daily_log = run_backtest(ind, cfg)

    summary = compute_metrics(daily_log, trades, cfg)
    summary["universe_symbols"] = int(closes.shape[1])
    summary["config"] = {k: getattr(cfg, k) for k in cfg.__dataclass_fields__}

    out_dir = Path(args.out)
    save_outputs(out_dir, trades, daily_log, summary, closes)

    print("\n" + "=" * 68)
    print(f" TURTLE CRYPTO BACKTEST  —  {summary['start']}  →  {summary['end']}")
    print("=" * 68)
    for k in (
        "starting_capital", "ending_equity", "total_return", "cagr", "ann_vol",
        "sharpe", "sortino", "max_drawdown", "calmar", "num_trades", "win_rate",
        "profit_factor", "avg_hold_days", "exits_stop", "exits_donchian",
        "avg_positions", "max_positions", "avg_heat_pct", "max_heat_pct",
    ):
        v = summary.get(k)
        if isinstance(v, float):
            if k in ("cagr", "ann_vol", "max_drawdown", "avg_heat_pct",
                     "max_heat_pct", "win_rate", "total_return"):
                print(f" {k:>22s}: {v*100:>8.2f}%")
            else:
                print(f" {k:>22s}: {v:>10.3f}")
        else:
            print(f" {k:>22s}: {v}")
    print("=" * 68)
    print(f" artefacts → {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
