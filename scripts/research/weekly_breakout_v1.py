#!/usr/bin/env python
"""
Weekly Breakout Trend Filter Strategy — v1 reference implementation.

Spec summary:
  - Universe:      liquid Coinbase USDC spot pairs (USD pairs are deprecated on Advanced),
                   excluding stablecoins, LSTs, illiquid pairs (90d median $-vol < $1M),
                   insufficient history (<365d), or poor coverage (<90%).
  - Frequency:     daily bars; rebalance weekly on Monday (or first available trading day).
                   Signals computed at prior-day close, executed at next open (one-bar lag).
  - Indicators:    MA(5), MA(40), 5-day breakout high/low, 20-day ATR, 40-day realized vol,
                   20/40/90-day momentum, composite momentum score (mean of cross-sectional
                   rank-percentiles, scaled 0-100).
  - Entry:         price > prior 5-day high  AND  MA(5) > MA(40)  AND  mom_score >= 40
                   AND in eligible (live + liquid + clean) universe.
  - Selection:     top 20 by momentum score.
  - Exits (any):   MA(5) < MA(40), close < prior 5-day low, mom_score < 40,
                   fell out of top-20, lost eligibility, ATR stop hit (entry - 3 * 20d ATR).
  - Sizing:        inverse-vol weights (1/vol40), capped at 15% per asset, gross 100%
                   (or fewer if fewer eligible; cash if zero).
  - Costs:         25 bps fee + 5 bps slippage = 30 bps per side (60 bps RT).
  - Stops:         fixed ATR stop checked intra-bar via daily low.

Writes to artifacts/research/weekly_breakout_v1/.
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import duckdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Settings ───────────────────────────────────────────────────────────
LAKE = "/Users/russellfloyd/Dropbox/NRT/nrt_dev/data/coinbase_crypto_ohlcv_lake.duckdb"
OUT  = Path("artifacts/research/weekly_breakout_v1")
FIG  = OUT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

ANN     = 365.0
INITIAL = 100_000.0

# Universe filters
MIN_HISTORY_DAYS  = 365
MIN_COVERAGE      = 0.90
LIQ_MIN_USD       = 500_000.0     # 90-day median $-volume (institutionally relevant; sensitivity at $1M / $5M)
MIN_ELIGIBLE_AT_START = 15        # backtest only starts when eligible-universe size first hits this
STABLES = {
    "USDT","DAI","USDP","PAX","EURC","TUSD","PYUSD","USDS","FRAX","USDD","GUSD",
    "CBETH","MSOL","LSETH","OETH","WSTETH",  # LSTs that track an underlying
}

# Strategy parameters
MA_FAST       = 5
MA_SLOW       = 40
BO_WIN        = 5
ATR_WIN       = 20
ATR_STOP_MULT = 3.0
VOL_WIN       = 40
MOM_WINS      = (20, 40, 90)
MOM_FLOOR     = 40.0
MAX_POSITIONS = 20
MAX_WEIGHT    = 0.15

# Costs (per side)
FEE_BPS       = 25.0
SLIPPAGE_BPS  = 5.0
COST_PER_SIDE = (FEE_BPS + SLIPPAGE_BPS) / 10000.0   # 30 bps one-side


# ── Universe loading ──────────────────────────────────────────────────
def load_universe(restrict_to: list[str] | None = None) -> tuple[list[str], dict]:
    con = duckdb.connect(LAKE, read_only=True)
    syms_df = con.execute(
        "SELECT symbol, MIN(ts) AS first_ts, MAX(ts) AS last_ts, COUNT(*) AS n_days "
        "FROM bars_1d_clean WHERE symbol LIKE '%-USDC' GROUP BY symbol"
    ).df()
    syms_df["first_ts"] = pd.to_datetime(syms_df["first_ts"], utc=True)
    syms_df["last_ts"]  = pd.to_datetime(syms_df["last_ts"], utc=True)
    syms_df["span_days"] = (syms_df["last_ts"] - syms_df["first_ts"]).dt.days
    syms_df["coverage"]  = syms_df["n_days"] / syms_df["span_days"].replace(0, np.nan)
    syms_df["base"]      = syms_df["symbol"].str.split("-").str[0]
    keep = syms_df[
        (~syms_df["base"].isin(STABLES))
        & (syms_df["span_days"] >= MIN_HISTORY_DAYS)
        & (syms_df["coverage"]  >= MIN_COVERAGE)
    ].copy()
    syms = sorted(keep["symbol"].tolist())
    if restrict_to is not None:
        restrict_set = set(restrict_to)
        syms = [s for s in syms if s in restrict_set]
        print(f"[load_universe] {len(syms)} pairs after restrict_to (intersection of curated set + structural filters)")
    else:
        print(f"[load_universe] {len(syms)} pre-filter pairs (history+coverage+stablecoin filter)")

    bars: dict[str, pd.DataFrame] = {}
    for s in syms:
        b = con.execute(
            "SELECT ts, open, high, low, close, volume "
            f"FROM bars_1d_clean WHERE symbol='{s}' ORDER BY ts"
        ).df()
        if b.empty:
            continue
        b["ts"] = pd.to_datetime(b["ts"]).dt.tz_convert("UTC").dt.tz_localize(None)
        b = b.set_index("ts").sort_index()
        bars[s] = b
    con.close()
    return syms, bars


def assemble_panels(syms, bars) -> dict[str, pd.DataFrame]:
    O = pd.DataFrame({s: bars[s]["open"]   for s in syms}).sort_index()
    H = pd.DataFrame({s: bars[s]["high"]   for s in syms}).sort_index()
    L = pd.DataFrame({s: bars[s]["low"]    for s in syms}).sort_index()
    C = pd.DataFrame({s: bars[s]["close"]  for s in syms}).sort_index()
    V = pd.DataFrame({s: bars[s]["volume"] for s in syms}).sort_index()
    # 90-day median dollar volume
    DV = (V * C).rolling(90).median()
    return dict(O=O, H=H, L=L, C=C, V=V, DV=DV)


# ── Indicators ────────────────────────────────────────────────────────
def compute_indicators(P: dict, liq_min_usd: float) -> dict:
    O, H, L, C, DV = P["O"], P["H"], P["L"], P["C"], P["DV"]
    ma_f = C.rolling(MA_FAST).mean()
    ma_s = C.rolling(MA_SLOW).mean()
    # Prior 5-day high/low — shift by 1 so "today's close > prior 5-day high"
    bo_high = C.rolling(BO_WIN).max().shift(1)
    bo_low  = C.rolling(BO_WIN).min().shift(1)
    # ATR: mean of true range over ATR_WIN
    prev_close = C.shift(1)
    tr1 = H - L
    tr2 = (H - prev_close).abs()
    tr3 = (L - prev_close).abs()
    tr  = pd.concat([tr1.stack(), tr2.stack(), tr3.stack()], axis=1).max(axis=1).unstack()
    atr = tr.rolling(ATR_WIN).mean()
    # 40-day annualized realized vol (close-to-close)
    ret = C.pct_change(fill_method=None)
    vol40 = ret.rolling(VOL_WIN).std() * np.sqrt(ANN)
    # Momentum windows
    mom = {w: C.pct_change(w, fill_method=None) for w in MOM_WINS}
    # Live + liquid mask
    live   = C.notna()
    liquid = DV >= liq_min_usd
    eligible_universe = live & liquid
    return dict(
        ma_f=ma_f, ma_s=ma_s, bo_high=bo_high, bo_low=bo_low,
        atr=atr, vol40=vol40, mom=mom, ret=ret,
        live=live, liquid=liquid, eligible_universe=eligible_universe,
    )


def momentum_score(mom: dict, eligible_mask: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional rank-percentile per day, averaged across horizons, scaled 0-100.
    Rank only over assets that are eligible (live + liquid + non-NaN mom).
    """
    parts = []
    for w, m in mom.items():
        masked = m.where(eligible_mask & m.notna())
        rk = masked.rank(axis=1, pct=True) * 100.0
        parts.append(rk)
    score = sum(parts) / len(parts)
    return score


# ── Backtest engine ───────────────────────────────────────────────────
def backtest(panels: dict, ind: dict, params: dict | None = None,
              extra_universe_mask: pd.DataFrame | None = None) -> dict:
    """Run the weekly breakout backtest.

    extra_universe_mask: optional boolean DataFrame aligned to panels['C'] (date x symbol).
        If supplied, an asset is only eligible on a date if mask is True. Used for
        deterministic dynamic-universe rules (e.g., top-N by trailing $-volume).
    """
    p = dict(
        cost_per_side=COST_PER_SIDE,
        max_positions=MAX_POSITIONS,
        max_weight=MAX_WEIGHT,
        atr_stop_mult=ATR_STOP_MULT,
        mom_floor=MOM_FLOOR,
        rebal_dow=0,  # Monday
        verbose=False,
        require_breakout_entry=True,   # if False, skip the close>prior5d_high filter
        use_atr_stop=True,             # if False, disable intra-bar ATR stops entirely
        trailing_stop=False,            # if True, stop = highest_close_since_entry - mult * entry_ATR
    )
    if params:
        p.update(params)

    O, H, L, C, DV = panels["O"], panels["H"], panels["L"], panels["C"], panels["DV"]
    ma_f, ma_s   = ind["ma_f"], ind["ma_s"]
    bo_high, bo_low = ind["bo_high"], ind["bo_low"]
    atr          = ind["atr"]
    vol40        = ind["vol40"]
    eligible_universe = ind["eligible_universe"]
    if extra_universe_mask is not None:
        # Align mask to panels then AND it in
        m = extra_universe_mask.reindex(index=C.index, columns=C.columns).fillna(False).astype(bool)
        eligible_universe = eligible_universe & m
    mom_score    = ind["mom_score"]
    ret          = ind["ret"]
    dates        = C.index

    # Two-tier eligibility:
    #   entry_eligible:  used to introduce NEW positions. Requires breakout (fresh upside trigger).
    #   hold_eligible:   used to evaluate retention. Drops if MA flips negative, mom < floor,
    #                    closes below prior 5d low (downside breakout / breakdown), or
    #                    falls out of universe. Does NOT require a fresh upside breakout
    #                    every week (since that would flush nearly the whole book each week).
    breakout_up  = (C > bo_high) & C.notna() & bo_high.notna()
    breakdown    = (C < bo_low)  & C.notna() & bo_low.notna()
    trend_pos    = (ma_f > ma_s) & ma_f.notna() & ma_s.notna()
    mom_ok       = mom_score >= p["mom_floor"]
    if p.get("require_breakout_entry", True):
        entry_eligible = breakout_up & trend_pos & mom_ok & eligible_universe
    else:
        entry_eligible = trend_pos & mom_ok & eligible_universe
    hold_eligible  = (~breakdown) & trend_pos & mom_ok & eligible_universe

    # Start when the eligible-universe is broad enough for top-N ranking to be meaningful.
    # We need both: (a) enough warm-up for all indicators, (b) ≥ MIN_ELIGIBLE_AT_START liquid+live names.
    n_warm = max(max(MOM_WINS), MA_SLOW, ATR_WIN, VOL_WIN) + 5
    elig_size = eligible_universe.sum(axis=1)
    min_n = p.get("min_eligible_at_start", MIN_ELIGIBLE_AT_START)
    candidate_starts = elig_size.index[(elig_size >= min_n) & (elig_size.index >= dates[n_warm])]
    if len(candidate_starts) == 0:
        start_date = dates[n_warm]
    else:
        start_date = candidate_starts[0]
    start_pos = dates.get_indexer([start_date], method="bfill")[0]

    # Rebalance schedule: every Monday on/after start_date
    rebal_dates = dates[(dates.dayofweek == p["rebal_dow"]) & (dates >= start_date)]
    rebal_set = set(rebal_dates)

    # State
    cash = INITIAL
    pos_shares: dict[str, float] = {}
    entry_price: dict[str, float] = {}
    entry_atr: dict[str, float] = {}
    entry_date: dict[str, pd.Timestamp] = {}
    highest_close: dict[str, float] = {}   # running max close since entry (for trailing stop)

    # History
    history = []          # daily (date, nav, n_pos, gross_exposure, cash_pct)
    trades = []           # trade tape
    rebal_log = []        # rebalance day decisions

    O_arr = O.values; H_arr = H.values; L_arr = L.values; C_arr = C.values
    atr_arr = atr.values
    col_idx = {s: i for i, s in enumerate(C.columns)}

    def asset_price_today(day_i, sym, field):
        i = col_idx[sym]
        return {"O": O_arr, "H": H_arr, "L": L_arr, "C": C_arr}[field][day_i, i]

    def mark_to_market(day_i):
        gross = 0.0
        for sym, sh in pos_shares.items():
            px = C_arr[day_i, col_idx[sym]]
            if np.isnan(px):
                # Fall back to last known close
                px = entry_price[sym]
            gross += sh * px
        return cash + gross, gross

    def stop_price(sym):
        """Compute current stop level for a held position."""
        if p.get("trailing_stop", False):
            ref = highest_close.get(sym, entry_price[sym])
        else:
            ref = entry_price[sym]
        return max(ref - p["atr_stop_mult"] * entry_atr[sym], 1e-9)

    def sell(day_i, sym, reason):
        nonlocal cash
        i = col_idx[sym]
        sh = pos_shares.pop(sym)
        # Determine exit price by reason:
        #   atr_stop  → stop level (if low <= stop, exit at stop price)
        #   other     → today's open (rebal day execution)
        if reason == "atr_stop":
            px = stop_price(sym)
        else:
            px = O_arr[day_i, i]
            if np.isnan(px):
                px = C_arr[day_i, i]
        notional = sh * px
        fee = abs(notional) * p["cost_per_side"]
        cash += notional - fee
        trades.append(dict(
            date=dates[day_i], symbol=sym, side="SELL", shares=sh, price=px,
            notional=notional, fee=fee, reason=reason,
        ))
        entry_price.pop(sym, None)
        entry_atr.pop(sym, None)
        entry_date.pop(sym, None)
        highest_close.pop(sym, None)

    def buy(day_i, sym, target_notional):
        nonlocal cash
        i = col_idx[sym]
        px = O_arr[day_i, i]
        if np.isnan(px) or px <= 0:
            return
        # Account for cost when sizing (so we don't accidentally go negative)
        gross_buy = target_notional / (1.0 + p["cost_per_side"])
        sh = gross_buy / px
        fee = gross_buy * p["cost_per_side"]
        cash -= (gross_buy + fee)
        pos_shares[sym] = pos_shares.get(sym, 0.0) + sh
        entry_price[sym] = px
        entry_atr[sym]   = atr_arr[day_i, i] if not np.isnan(atr_arr[day_i, i]) else 0.0
        entry_date[sym]  = dates[day_i]
        highest_close[sym] = px
        trades.append(dict(
            date=dates[day_i], symbol=sym, side="BUY", shares=sh, price=px,
            notional=gross_buy, fee=fee, reason="rebal_buy",
        ))

    def resize(day_i, sym, target_notional):
        """Adjust position to a new target notional at today's open."""
        nonlocal cash
        i = col_idx[sym]
        px = O_arr[day_i, i]
        if np.isnan(px) or px <= 0:
            return
        cur_sh = pos_shares.get(sym, 0.0)
        cur_notional = cur_sh * px
        delta_notional = target_notional - cur_notional
        if abs(delta_notional) < 1e-6:
            return
        if delta_notional > 0:
            # Buying more
            gross_buy = delta_notional / (1.0 + p["cost_per_side"])
            add_sh = gross_buy / px
            fee = gross_buy * p["cost_per_side"]
            cash -= (gross_buy + fee)
            pos_shares[sym] = cur_sh + add_sh
            trades.append(dict(date=dates[day_i], symbol=sym, side="BUY",
                                shares=add_sh, price=px,
                                notional=gross_buy, fee=fee, reason="resize_up"))
        else:
            # Selling some
            reduce_notional = -delta_notional
            sell_sh = reduce_notional / px
            sell_sh = min(sell_sh, cur_sh)
            proceeds = sell_sh * px
            fee = proceeds * p["cost_per_side"]
            cash += (proceeds - fee)
            pos_shares[sym] = cur_sh - sell_sh
            if pos_shares[sym] < 1e-9:
                pos_shares.pop(sym, None)
                entry_price.pop(sym, None)
                entry_atr.pop(sym, None)
                entry_date.pop(sym, None)
            trades.append(dict(date=dates[day_i], symbol=sym, side="SELL",
                                shares=sell_sh, price=px,
                                notional=-reduce_notional, fee=fee, reason="resize_down"))

    # ── Daily loop ────────────────────────────────────────────────────
    for day_i in range(start_pos, len(dates)):
        day = dates[day_i]

        # Step A: Intra-day ATR-stop check on existing positions (using today's low)
        if p.get("use_atr_stop", True):
            stops_today = []
            for sym, sh in list(pos_shares.items()):
                i = col_idx[sym]
                low_today = L_arr[day_i, i]
                sp = stop_price(sym)
                if not np.isnan(low_today) and low_today <= sp:
                    stops_today.append(sym)
            for sym in stops_today:
                sell(day_i, sym, "atr_stop")

        # Step B: If this is a rebalance day, execute the rebalance using today's data
        # (signals computed from yesterday's close, executed at today's open)
        is_rebal = day in rebal_set
        if is_rebal:
            # Yesterday's signal row
            yd = day_i - 1
            yest = dates[yd] if yd >= 0 else None

            # Score and eligibility from the prior close (no look-ahead)
            score_row  = mom_score.iloc[yd].copy()
            entry_row  = entry_eligible.iloc[yd].copy()
            hold_row   = hold_eligible.iloc[yd].copy()
            vol_row    = vol40.iloc[yd].copy()

            # Rank ALL hold-eligible assets by mom_score → top-N candidates
            hold_cands = score_row[hold_row & score_row.notna()].sort_values(ascending=False)
            top_n_hold = list(hold_cands.head(p["max_positions"]).index)

            # New-entry candidates: must be entry-eligible (breakout TODAY) AND in the top-N
            # of the hold-eligible ranking.
            entry_cands = score_row[entry_row & score_row.notna()].sort_values(ascending=False)
            entry_set   = set(entry_cands.index) & set(top_n_hold)

            # ── SELLS ─────────────────────────────────────────────────
            # Exit a held position if it has either:
            #   (a) fallen out of the top-N hold-eligible ranking, OR
            #   (b) failed hold-eligibility (MA flip, mom below floor, breakdown, lost universe)
            held = list(pos_shares.keys())
            sells = []
            for sym in held:
                if (sym not in top_n_hold) or (not bool(hold_row.get(sym, False))):
                    sells.append(sym)
            for sym in sells:
                sell(day_i, sym, "rebal_sell")

            # ── TARGET PORTFOLIO ──────────────────────────────────────
            # Retain held positions still in top-N (already kept above), plus add new entries
            # from entry_set up to MAX_POSITIONS total.
            retained = [s for s in top_n_hold if s in pos_shares]
            slots = p["max_positions"] - len(retained)
            new_entries = [s for s in top_n_hold
                           if s not in pos_shares and s in entry_set][:slots]
            target_syms = retained + new_entries

            # Now figure out cash + value held at today's open for sizing
            # value of retained positions at today's open
            held_after_sells = list(pos_shares.keys())
            value_held = 0.0
            for sym in held_after_sells:
                i = col_idx[sym]
                px = O_arr[day_i, i]
                if np.isnan(px):
                    px = C_arr[day_i, i]
                value_held += pos_shares[sym] * px
            equity = cash + value_held

            # Compute inverse-vol weights for target_syms
            if len(target_syms) > 0:
                vols = np.array([vol_row.get(s, np.nan) for s in target_syms], dtype=float)
                # Replace NaN/zero vols with median to be defensive
                med_vol = np.nanmedian(vols) if np.isfinite(np.nanmedian(vols)) else 1.0
                vols = np.where((np.isnan(vols)) | (vols <= 0), med_vol, vols)
                inv = 1.0 / vols
                w = inv / inv.sum()
                # Apply max-weight cap iteratively
                for _ in range(20):
                    over = w > p["max_weight"]
                    if not over.any():
                        break
                    excess = (w[over] - p["max_weight"]).sum()
                    w[over] = p["max_weight"]
                    others = ~over
                    if others.any() and w[others].sum() > 0:
                        w[others] += excess * w[others] / w[others].sum()
                    else:
                        break
                # Now scale to gross = 100% (only if there's enough capital;
                # if fewer than max_positions, gross = sum(w) which may already be 1.0)
                # We've already normalized to 1.0 above. So gross = 100% if N >= 1.
                target_notionals = {s: equity * w[i] for i, s in enumerate(target_syms)}
            else:
                target_notionals = {}

            # Apply: resize retained, buy new
            for sym, tn in target_notionals.items():
                if sym in pos_shares:
                    resize(day_i, sym, tn)
                else:
                    buy(day_i, sym, tn)

            # Log
            rebal_log.append(dict(
                date=day, n_target=len(target_syms), equity=equity, cash_after=cash,
                n_hold_cands=len(hold_cands), n_entry_cands=len(entry_cands),
                n_new_entries=len(new_entries),
                top_score=float(hold_cands.head(1).values[0]) if len(hold_cands) > 0 else np.nan,
                med_score=float(hold_cands.head(p["max_positions"]).median()) if len(hold_cands) > 0 else np.nan,
            ))

        # Step C: Update trailing-stop reference for held positions (highest close since entry)
        if p.get("trailing_stop", False):
            for sym in list(pos_shares.keys()):
                i = col_idx[sym]
                c_today = C_arr[day_i, i]
                if not np.isnan(c_today) and c_today > highest_close.get(sym, 0):
                    highest_close[sym] = c_today

        # Step D: Record EOD NAV (mark to today's close)
        nav, gross = mark_to_market(day_i)
        history.append(dict(
            date=day, nav=nav, n_pos=len(pos_shares),
            gross_exposure=gross / nav if nav > 0 else 0.0,
            cash=cash, cash_pct=cash / nav if nav > 0 else 1.0,
        ))

        if p["verbose"] and is_rebal:
            print(f"  {day.date()}: NAV ${nav:>12,.0f}  pos={len(pos_shares):>2d}  gross={gross/nav:>5.1%}")

    eq = pd.DataFrame(history).set_index("date")
    tr = pd.DataFrame(trades)
    rl = pd.DataFrame(rebal_log)
    return dict(equity=eq, trades=tr, rebal_log=rl)


# ── Metrics ───────────────────────────────────────────────────────────
def metrics_from_nav(nav: pd.Series, name: str = "") -> dict:
    nav = nav.dropna()
    if len(nav) < 2:
        return dict(name=name)
    r = nav.pct_change().dropna()
    yrs = len(nav) / ANN
    cagr = (nav.iloc[-1] / nav.iloc[0]) ** (1 / max(yrs, 1e-9)) - 1
    sh = r.mean() / r.std() * np.sqrt(ANN) if r.std() > 0 else 0.0
    dd = float((nav / nav.cummax() - 1).min())
    total = nav.iloc[-1] / nav.iloc[0] - 1
    return dict(name=name, cagr=cagr, vol=r.std()*np.sqrt(ANN),
                sharpe=sh, max_dd=dd, total=total,
                years=yrs, final=nav.iloc[-1])


def compute_benchmarks(panels: dict, ind: dict, start_date) -> dict:
    """BTC-USDC and an MA(5/40) L1+L2+DeFi-ish basket for comparison."""
    C = panels["C"]
    O = panels["O"]
    # BTC-USDC B&H from start_date
    if "BTC-USDC" in C.columns:
        btc_o = O["BTC-USDC"]; btc_c = C["BTC-USDC"]
        btc_o, btc_c = btc_o.align(btc_c, join="inner")
        btc_o = btc_o.loc[btc_o.index >= start_date]
        btc_c = btc_c.loc[btc_c.index >= start_date]
        # Apply 30 bps one-side buy cost at inception
        btc_ret = (btc_c / btc_o - 1).fillna(0.0)
        btc_nav = INITIAL * (1 + btc_ret).cumprod() * (1 - COST_PER_SIDE)
    else:
        btc_nav = pd.Series(dtype=float)
    return dict(btc_bh=btc_nav)


# ── Reporting / figures ───────────────────────────────────────────────
def write_figures(result: dict, panels: dict, ind: dict, out_dir: Path,
                   label: str = "Weekly Breakout v1"):
    eq = result["equity"]
    nav = eq["nav"]
    bench = compute_benchmarks(panels, ind, eq.index[0])
    btc = bench["btc_bh"]

    m_strat = metrics_from_nav(nav, label)
    m_btc   = metrics_from_nav(btc,  "BTC-USDC B&H")

    # Figure 1: equity curve + drawdown
    fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True,
                              gridspec_kw={"height_ratios": [3, 1, 1]})
    ax = axes[0]
    ax.plot(nav.index, nav/1e3,
             label=f"{label}  (Sh={m_strat['sharpe']:.2f}, DD={m_strat['max_dd']:.0%}, CAGR={m_strat['cagr']*100:.0f}%)",
             color="#1f77b4", lw=2)
    if not btc.empty:
        # Align starts for fair comparison
        common_start = max(nav.index[0], btc.index[0])
        n2 = nav.loc[nav.index >= common_start]
        b2 = btc.loc[btc.index >= common_start]
        # Re-base to $100k at common start
        n2_r = n2 / n2.iloc[0] * INITIAL
        b2_r = b2 / b2.iloc[0] * INITIAL
        ax.plot(b2_r.index, b2_r/1e3,
                label=f"BTC-USDC B&H  (Sh={m_btc['sharpe']:.2f}, DD={m_btc['max_dd']:.0%}, CAGR={m_btc['cagr']*100:.0f}%)",
                color="#d62728", lw=1.6, ls="--", alpha=0.85)
    ax.set_yscale("log"); ax.set_ylabel("NAV ($k, log)")
    ax.set_title(f"{label} — equity curve  (initial $100k)")
    ax.legend(loc="lower right"); ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.fill_between(eq.index, eq["n_pos"], 0, color="#1f77b4", alpha=0.45, label="# positions")
    ax.set_ylabel("# positions"); ax.grid(True, alpha=0.3); ax.legend(loc="upper left")

    ax = axes[2]
    dd = (nav / nav.cummax() - 1) * 100
    ax.fill_between(dd.index, dd, 0, color="#1f77b4", alpha=0.5, label=f"{label}")
    if not btc.empty:
        dd_b = (btc / btc.cummax() - 1) * 100
        ax.plot(dd_b.index, dd_b, color="#d62728", lw=1.0, ls="--", label="BTC-USDC B&H")
    ax.axhline(0, color="k", lw=0.5); ax.set_ylabel("DD (%)")
    ax.legend(loc="lower left", fontsize=9); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir/"01_equity_drawdown.png", dpi=110, bbox_inches="tight")
    plt.close()

    # Figure 2: gross exposure + cash %
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    ax = axes[0]
    ax.plot(eq.index, eq["gross_exposure"]*100, color="#1f77b4", lw=1.2, label="Gross exposure %")
    ax.axhline(100, color="k", lw=0.5, ls="--", alpha=0.5)
    ax.set_ylabel("Gross exposure (%)"); ax.legend(); ax.grid(True, alpha=0.3)
    ax.set_title(f"{label} — exposure & cash over time")
    ax = axes[1]
    ax.plot(eq.index, eq["cash_pct"]*100, color="#2ca02c", lw=1.2, label="Cash %")
    ax.fill_between(eq.index, eq["cash_pct"]*100, 0, color="#2ca02c", alpha=0.2)
    ax.set_ylabel("Cash (%)"); ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir/"02_exposure_and_cash.png", dpi=110, bbox_inches="tight")
    plt.close()

    # Figure 3: calendar-year returns
    yrly = nav.resample("YE").last()
    yrly_ret = yrly.pct_change().dropna()
    # Prepend first-year partial (from start to first year-end)
    first_year = nav.index[0].year
    first_full = pd.Timestamp(f"{first_year}-12-31")
    nav_first = nav.iloc[0]
    if nav.index[0] <= first_full <= nav.index[-1]:
        nav_first_full = nav.loc[:first_full].iloc[-1]
        yrly_ret[pd.Timestamp(f"{first_year}-12-31")] = nav_first_full/nav_first - 1
        yrly_ret = yrly_ret.sort_index()

    btc_yrly = btc.resample("YE").last() if not btc.empty else pd.Series(dtype=float)
    btc_yrly_ret = btc_yrly.pct_change().dropna() if not btc_yrly.empty else pd.Series(dtype=float)
    if not btc.empty and pd.Timestamp(f"{first_year}-12-31") in btc.loc[:first_full].index:
        btc_first_full = btc.loc[:first_full].iloc[-1]
        btc_yrly_ret[pd.Timestamp(f"{first_year}-12-31")] = btc_first_full/btc.iloc[0] - 1
        btc_yrly_ret = btc_yrly_ret.sort_index()

    yrly_ret.index = yrly_ret.index.year
    btc_yrly_ret.index = btc_yrly_ret.index.year if not btc_yrly_ret.empty else btc_yrly_ret.index
    common = sorted(set(yrly_ret.index) & set(btc_yrly_ret.index)) if not btc_yrly_ret.empty else list(yrly_ret.index)

    fig, ax = plt.subplots(figsize=(13, 5))
    x = np.arange(len(common)); w = 0.4
    s_vals = [yrly_ret.get(y, np.nan)*100 for y in common]
    b_vals = [btc_yrly_ret.get(y, np.nan)*100 for y in common] if not btc_yrly_ret.empty else [0]*len(common)
    ax.bar(x - w/2, s_vals, w, label=label, color="#1f77b4")
    if not btc_yrly_ret.empty:
        ax.bar(x + w/2, b_vals, w, label="BTC-USDC B&H", color="#d62728")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xticks(x); ax.set_xticklabels([str(y) for y in common])
    ax.set_ylabel("Calendar-year return (%)")
    ax.set_title(f"{label} — calendar-year returns vs BTC B&H")
    ax.legend(); ax.grid(True, alpha=0.3, axis="y")
    for xi, sv in enumerate(s_vals):
        if not np.isnan(sv):
            ax.text(xi - w/2, sv + (3 if sv>=0 else -7), f"{sv:+.0f}%",
                    ha="center", fontsize=7, color="#1f77b4")
    for xi, bv in enumerate(b_vals):
        if not np.isnan(bv):
            ax.text(xi + w/2, bv + (3 if bv>=0 else -7), f"{bv:+.0f}%",
                    ha="center", fontsize=7, color="#d62728")
    plt.tight_layout()
    plt.savefig(out_dir/"03_calendar_year.png", dpi=110, bbox_inches="tight")
    plt.close()

    # Figure 4: position turnover (number of distinct symbols held per quarter)
    if not result["trades"].empty:
        tr = result["trades"].copy()
        tr["q"] = pd.PeriodIndex(tr["date"], freq="Q")
        unique_per_q = tr.groupby("q")["symbol"].nunique()
        fig, ax = plt.subplots(figsize=(13, 4))
        ax.bar([str(p) for p in unique_per_q.index], unique_per_q.values, color="#9467bd", alpha=0.7)
        ax.set_xlabel("Quarter"); ax.set_ylabel("Distinct symbols traded")
        ax.set_title(f"{label} — universe diversity per quarter")
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        plt.savefig(out_dir/"04_quarterly_diversity.png", dpi=110, bbox_inches="tight")
        plt.close()

    # Save metrics
    summary = dict(
        strategy=m_strat,
        btc_bh=m_btc,
        n_rebalances=len(result["rebal_log"]),
        n_trades=len(result["trades"]),
        median_positions=float(eq["n_pos"].median()),
        median_gross_exposure=float(eq["gross_exposure"].median()),
        median_cash_pct=float(eq["cash_pct"].median()),
        start=str(eq.index[0].date()),
        end=str(eq.index[-1].date()),
    )
    with open(out_dir/"summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    return summary


# ── Main ──────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    print(f"=== Weekly Breakout v1 ===\n")
    syms, bars = load_universe()
    panels = assemble_panels(syms, bars)
    print(f"[panels] {len(panels['C'].columns)} cols × {len(panels['C'])} rows  "
          f"({panels['C'].index[0].date()} -> {panels['C'].index[-1].date()})")
    ind = compute_indicators(panels, LIQ_MIN_USD)
    ind["mom_score"] = momentum_score(ind["mom"], ind["eligible_universe"])
    # Snapshot: count of liquid+live names over time
    n_liquid = ind["eligible_universe"].sum(axis=1)
    print(f"[liquidity] median liquid+live universe size: {int(n_liquid.median())}  "
          f"(min {n_liquid.min()}, max {n_liquid.max()})")
    # Identify the first date where universe is broad enough
    first_broad = n_liquid.index[n_liquid >= MIN_ELIGIBLE_AT_START]
    if len(first_broad) > 0:
        print(f"[liquidity] first day with ≥{MIN_ELIGIBLE_AT_START} eligible names: {first_broad[0].date()}")

    print(f"\n[backtest] running...")
    result = backtest(panels, ind)
    print(f"[backtest] done in {time.time()-t0:.1f}s")

    eq = result["equity"]; tr = result["trades"]; rl = result["rebal_log"]
    eq.to_csv(OUT/"equity.csv")
    tr.to_csv(OUT/"trades.csv", index=False)
    rl.to_csv(OUT/"rebal_log.csv", index=False)
    print(f"  Rebalances: {len(rl)}    Trades: {len(tr)}")
    print(f"  Period: {eq.index[0].date()} -> {eq.index[-1].date()}  "
          f"({(eq.index[-1]-eq.index[0]).days/365:.1f} years)")

    nav = eq["nav"]
    m_strat = metrics_from_nav(nav, "Weekly Breakout v1")
    print(f"\n[metrics]")
    print(f"  Final NAV:  ${nav.iloc[-1]:>12,.0f}  (start ${INITIAL:,.0f})")
    print(f"  CAGR:       {m_strat['cagr']*100:>+7.1f}%")
    print(f"  Vol:        {m_strat['vol']*100:>7.1f}%")
    print(f"  Sharpe:     {m_strat['sharpe']:>+7.2f}")
    print(f"  MaxDD:      {m_strat['max_dd']*100:>+7.1f}%")
    print(f"  Total:      {m_strat['total']*100:>+7.0f}%")
    print(f"  Median pos: {eq['n_pos'].median():.1f}")
    print(f"  Median gross exposure: {eq['gross_exposure'].median()*100:.1f}%")
    print(f"  Median cash %:         {eq['cash_pct'].median()*100:.1f}%")

    bench = compute_benchmarks(panels, ind, eq.index[0])
    btc = bench["btc_bh"]
    if not btc.empty:
        m_btc = metrics_from_nav(btc, "BTC-USDC B&H")
        print(f"\n[benchmark BTC-USDC B&H, same period]")
        print(f"  Final NAV:  ${btc.iloc[-1]:>12,.0f}")
        print(f"  CAGR:       {m_btc['cagr']*100:>+7.1f}%")
        print(f"  Sharpe:     {m_btc['sharpe']:>+7.2f}")
        print(f"  MaxDD:      {m_btc['max_dd']*100:>+7.1f}%")
        print(f"  Total:      {m_btc['total']*100:>+7.0f}%")

    summary = write_figures(result, panels, ind, FIG)
    print(f"\n[done] wrote {OUT}")


if __name__ == "__main__":
    main()
