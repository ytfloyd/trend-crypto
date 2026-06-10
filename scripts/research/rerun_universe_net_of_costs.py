#!/usr/bin/env python
"""
Re-run the complete MA(5/40) USDC universe study with 20 bps round-trip
transaction costs applied to per-position turnover. Mid-rate entry / no
slippage assumption.

Cost model:
  r_net_t = r_gross_t  -  turnover_t * cost_rate

where:
  - For per-pair strategies, turnover_t = |sig_t - sig_{t-1}|
  - For basket strategies, turnover_t = sum_i |w_i,t - w_i,t-1|
  - cost_rate = 0.0020 (20 bps round-trip per unit of weight change)

The convention treats one full unit of weight change (e.g., 0 -> 1 or 1 -> 0)
as one half-round-trip. So a complete in-and-out cycle (0 -> 1 -> 0) costs
2 * cost_rate / 2 = cost_rate = 20 bps. This matches the standard "round-trip"
quote convention for fee schedules.

Overwrites:
  - artifacts/research/ma_5_40_usdc_universe/usdc_universe_ma_5_40_results.csv
  - artifacts/research/ma_5_40_usdc_universe/usdc_universe_ma_5_40_results_with_category.csv
  - artifacts/research/ma_5_40_usdc_universe/category_stats.csv
  - artifacts/research/ma_5_40_usdc_universe/wfo_per_pair_results.csv
  - artifacts/research/ma_5_40_usdc_universe/wfo_oos_returns_by_pair.pkl
  - artifacts/research/ma_5_40_usdc_universe/basket_returns.parquet
  - artifacts/research/ma_5_40_usdc_universe/basket_wfo_returns.parquet
  - artifacts/research/ma_5_40_usdc_universe/high_quality_basket_returns.parquet
  - artifacts/research/ma_5_40_usdc_universe/highq_basket_wfo_returns.parquet
  - artifacts/research/ma_5_40_usdc_universe/highq_basket_wfo_selections.csv
  - artifacts/research/ma_5_40_usdc_universe/figures/*.png  (regenerated)

Preserves the gross-of-cost data files in a sibling `_gross_backup/` folder.
"""
from __future__ import annotations

import pickle
import shutil
import time
from pathlib import Path

import duckdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Settings ───────────────────────────────────────────────────────────
COST_BPS_ROUND_TRIP = 20         # 20 bps round-trip per unit of position change
COST_RATE = COST_BPS_ROUND_TRIP / 10000.0
LAKE = "/Users/russellfloyd/Dropbox/NRT/nrt_dev/data/coinbase_crypto_ohlcv_lake.duckdb"
ANN = 365.0
INITIAL = 100_000.0

OUT = Path("artifacts/research/ma_5_40_usdc_universe")
FIG = OUT / "figures"
BACKUP = OUT / "_gross_backup"
FIG.mkdir(parents=True, exist_ok=True)
BACKUP.mkdir(parents=True, exist_ok=True)

# ── Common helpers ────────────────────────────────────────────────────
def stats(r: pd.Series, label: str = "") -> dict:
    r = pd.Series(r).dropna()
    if len(r) == 0 or r.std() == 0:
        return dict(label=label, cagr=0.0, vol=0.0, sharpe=0.0,
                    max_dd=0.0, total=0.0, nav=pd.Series(dtype=float))
    nav = INITIAL * (1 + r).cumprod()
    yrs = len(r) / ANN
    return dict(
        label=label,
        cagr=(nav.iloc[-1] / INITIAL) ** (1 / max(yrs, 1e-6)) - 1,
        vol=r.std() * np.sqrt(ANN),
        sharpe=r.mean() / r.std() * np.sqrt(ANN),
        max_dd=float((nav / nav.cummax() - 1).min()),
        total=nav.iloc[-1] / INITIAL - 1,
        nav=nav,
        ret=r,
    )


def per_pair_stats_block(opens: pd.Series, closes: pd.Series,
                          fast: int, slow: int) -> tuple[dict, dict, pd.Series, pd.Series]:
    """Return (strat_stats, bh_stats, strat_returns_net, bh_returns)."""
    r = (closes / opens - 1).fillna(0.0)
    ma_f = closes.rolling(fast).mean()
    ma_s = closes.rolling(slow).mean()
    sig = (ma_f > ma_s).astype(float).where(closes.notna(), 0.0).shift(1).fillna(0.0)
    turnover = sig.diff().abs().fillna(0.0)
    r_strat = r * sig - turnover * COST_RATE
    return stats(r_strat), stats(r), r_strat, r


# Read the universe of symbols + their bars
def load_universe() -> tuple[list[str], pd.DataFrame, pd.DataFrame]:
    con = duckdb.connect(LAKE, read_only=True)
    syms_df = con.execute(
        "SELECT symbol, MIN(ts) AS first_ts, MAX(ts) AS last_ts, COUNT(*) AS n_days "
        "FROM bars_1d_clean WHERE symbol LIKE '%-USDC' GROUP BY symbol"
    ).df()
    syms_df["first_ts"] = pd.to_datetime(syms_df["first_ts"], utc=True)
    syms_df["last_ts"]  = pd.to_datetime(syms_df["last_ts"], utc=True)
    syms_df["span_days"] = (syms_df["last_ts"] - syms_df["first_ts"]).dt.days
    syms_df["coverage"]  = syms_df["n_days"] / syms_df["span_days"].replace(0, np.nan)
    keep = syms_df[(syms_df["span_days"] >= 365 * 3) & (syms_df["coverage"] >= 0.90)].copy()
    syms = sorted(keep["symbol"].tolist())
    print(f"[load_universe] {len(syms)} eligible USDC pairs (>=3y history, >=90% coverage)")

    # Load O/C panels
    opens, closes = {}, {}
    for s in syms:
        b = con.execute(
            f"SELECT ts, open, close FROM bars_1d_clean WHERE symbol='{s}' ORDER BY ts"
        ).df()
        if b.empty:
            continue
        b["ts"] = pd.to_datetime(b["ts"]).dt.tz_convert("UTC").dt.tz_localize(None)
        b = b.set_index("ts").sort_index()
        opens[s]  = b["open"]
        closes[s] = b["close"]
    con.close()
    O = pd.DataFrame(opens).sort_index()
    C = pd.DataFrame(closes).sort_index()
    return syms, O, C


# ── Step 1: Per-pair universe study (186 pairs) ───────────────────────
def step1_universe(syms, O, C) -> pd.DataFrame:
    print(f"\n[step1] Per-pair universe (n={len(syms)}) — net of {COST_BPS_ROUND_TRIP} bps...")
    rows = []
    for s in syms:
        opens = O[s].dropna()
        closes = C[s].dropna()
        opens, closes = opens.align(closes, join="inner")
        if len(opens) < 365:
            continue
        s_strat, s_bh, _, _ = per_pair_stats_block(opens, closes, 5, 40)
        years = len(opens) / ANN
        # pct_long
        ma_f = closes.rolling(5).mean(); ma_s = closes.rolling(40).mean()
        sig = (ma_f > ma_s).astype(float).where(closes.notna(), 0.0).shift(1).fillna(0.0)
        pct_long = float(sig.mean())
        rows.append(dict(
            symbol=s, years=years, n_days=len(opens), pct_long=pct_long,
            strat_cagr=s_strat["cagr"], strat_sharpe=s_strat["sharpe"],
            strat_maxdd=s_strat["max_dd"], strat_total=s_strat["total"], strat_vol=s_strat["vol"],
            bh_cagr=s_bh["cagr"], bh_sharpe=s_bh["sharpe"],
            bh_maxdd=s_bh["max_dd"], bh_total=s_bh["total"], bh_vol=s_bh["vol"],
            edge_x=(1+s_strat["total"]) / (1+s_bh["total"]) if s_bh["total"] > -1 else np.nan,
            edge_cagr_pp=(s_strat["cagr"] - s_bh["cagr"]) * 100,
            edge_sharpe=s_strat["sharpe"] - s_bh["sharpe"],
            edge_maxdd_pp=(s_strat["max_dd"] - s_bh["max_dd"]) * 100,
        ))
    df = pd.DataFrame(rows)
    print(f"[step1] {len(df)} pairs analyzed")
    # Save
    df.to_csv(OUT / "usdc_universe_ma_5_40_results.csv", index=False)
    print(f"        wrote {OUT / 'usdc_universe_ma_5_40_results.csv'}")
    return df


# ── Step 2: Universe-level figures (scatter, distributions, segmented) ─
def step2_universe_figures(df: pd.DataFrame):
    print(f"\n[step2] Regenerating universe-level figures...")

    # Figure 01: scatter CAGR + MaxDD
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    ax = axes[0]
    ax.scatter(df["bh_cagr"]*100, df["strat_cagr"]*100, s=22, alpha=0.65, c="#1f77b4")
    lo = min(df["bh_cagr"].min(), df["strat_cagr"].min())*100 - 5
    hi = max(df["bh_cagr"].max(), df["strat_cagr"].max())*100 + 5
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.6, label="y = x")
    # Annotate key names
    for _, row in df.iterrows():
        if row["symbol"] in {"BTC-USDC","ETH-USDC","SOL-USDC","LINK-USDC","DOGE-USDC","SUI-USDC","RNDR-USDC"}:
            ax.annotate(row["symbol"].replace("-USDC",""),
                        (row["bh_cagr"]*100, row["strat_cagr"]*100),
                        fontsize=8, alpha=0.8, xytext=(4, 2), textcoords="offset points")
    ax.set_xlabel("B&H CAGR (%)"); ax.set_ylabel("Strategy CAGR (%)")
    ax.set_title(f"Strategy CAGR vs B&H CAGR (net of {COST_BPS_ROUND_TRIP} bps round-trip)")
    ax.grid(True, alpha=0.3); ax.legend()

    ax = axes[1]
    ax.scatter(df["bh_maxdd"]*100, df["strat_maxdd"]*100, s=22, alpha=0.65, c="#d62728")
    lo = min(df["bh_maxdd"].min(), df["strat_maxdd"].min())*100 - 5
    hi = 5
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.6, label="y = x")
    ax.set_xlabel("B&H MaxDD (%)"); ax.set_ylabel("Strategy MaxDD (%)")
    ax.set_title(f"Strategy MaxDD vs B&H MaxDD")
    ax.grid(True, alpha=0.3); ax.legend()
    plt.tight_layout()
    plt.savefig(FIG/"01_scatter_cagr_and_maxdd.png", dpi=110, bbox_inches="tight")
    plt.close()

    # Figure 02: edge distributions
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    ax = axes[0,0]
    ax.hist(np.clip(df["edge_x"], 0.01, 100), bins=40, color="#1f77b4", alpha=0.7, edgecolor="white")
    ax.axvline(1.0, color="k", lw=0.7, label="Edge = 1x (no edge)")
    ax.axvline(df["edge_x"].median(), color="r", lw=0.7, ls="--", label=f"Median = {df['edge_x'].median():.2f}x")
    ax.set_xscale("log"); ax.set_xlabel("Edge ratio (strategy / B&H)"); ax.set_ylabel("Pairs")
    ax.set_title("Edge ratio distribution (log scale, clipped 0.01-100)")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[0,1]
    ax.hist(df["edge_sharpe"], bins=40, color="#9467bd", alpha=0.7, edgecolor="white")
    ax.axvline(0, color="k", lw=0.7, label="Zero edge")
    ax.axvline(df["edge_sharpe"].median(), color="r", lw=0.7, ls="--", label=f"Median = {df['edge_sharpe'].median():+.2f}")
    ax.set_xlabel("Sharpe edge (strategy - B&H)"); ax.set_ylabel("Pairs")
    pos_pct = (df["edge_sharpe"] > 0).mean()*100
    ax.set_title(f"Sharpe edge distribution\n{pos_pct:.0f}% of pairs have positive edge")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1,0]
    ax.hist(df["edge_maxdd_pp"], bins=40, color="#2ca02c", alpha=0.7, edgecolor="white")
    ax.axvline(0, color="k", lw=0.7, label="Zero edge")
    ax.axvline(df["edge_maxdd_pp"].median(), color="r", lw=0.7, ls="--",
                label=f"Median = +{df['edge_maxdd_pp'].median():.1f} pp")
    ax.set_xlabel("MaxDD edge (pp, positive = strategy DD less severe)"); ax.set_ylabel("Pairs")
    pp = ((df["strat_maxdd"]-df["bh_maxdd"]) > 0).mean()*100
    ax.set_title(f"MaxDD edge distribution\n{pp:.0f}% of pairs have shallower DD")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1,1]
    df_c = df.copy()
    df_c["bh_outcome"] = pd.cut(df_c["bh_total"], bins=[-1.01, -0.9, -0.5, 0.5, 100],
                                  labels=["B&H wipeout (<-90%)", "B&H bear (-50/-90%)",
                                          "B&H neutral (-50/+50%)", "B&H winner (>+50%)"])
    for label, color in [("B&H winner (>+50%)", "#2ca02c"),
                         ("B&H neutral (-50/+50%)", "#1f77b4"),
                         ("B&H bear (-50/-90%)", "#ff7f0e"),
                         ("B&H wipeout (<-90%)", "#d62728")]:
        d = df_c[df_c["bh_outcome"]==label]
        ax.scatter(d["years"], d["strat_cagr"]*100, label=f"{label} (n={len(d)})",
                    alpha=0.7, s=22, color=color)
    ax.set_xlabel("Years of history"); ax.set_ylabel("Strategy CAGR (%)")
    ax.set_title("Strategy CAGR by history length")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG/"02_edge_distributions.png", dpi=110, bbox_inches="tight")
    plt.close()

    # Figure 03: segmented analysis
    df_c["bh_outcome"] = pd.cut(df["bh_total"], bins=[-1.01, -0.9, -0.5, 0.5, 100],
                                  labels=["wipeout", "bear", "neutral", "winner"])
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    seg = df_c.groupby("bh_outcome").agg(
        bh_total=("bh_total", "median"),
        strat_total=("strat_total", "median"),
        n=("symbol", "count"),
        wr=("edge_x", lambda x: (x>1).mean()*100),
    ).reset_index()
    seg = seg.iloc[::-1].reset_index(drop=True)  # winner first
    x = np.arange(len(seg))
    ax = axes[0]
    w = 0.4
    ax.bar(x - w/2, seg["strat_total"]*100, w, label="Strategy median total return", color="#1f77b4")
    ax.bar(x + w/2, seg["bh_total"]*100,    w, label="B&H median total return",      color="#d62728")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xticks(x); ax.set_xticklabels([f"{s}\n(n={n})" for s, n in zip(seg["bh_outcome"], seg["n"])])
    ax.set_ylabel("Total return (%)")
    ax.set_title(f"Median total return by B&H outcome segment (net of {COST_BPS_ROUND_TRIP} bps)")
    ax.legend(); ax.grid(True, alpha=0.3, axis="y")
    ax = axes[1]
    ax.bar(x, seg["wr"], color="#9467bd")
    ax.axhline(50, color="k", lw=0.5, ls="--")
    ax.set_xticks(x); ax.set_xticklabels([f"{s}\n(n={n})" for s, n in zip(seg["bh_outcome"], seg["n"])])
    ax.set_ylabel("Strategy win-rate (%)")
    ax.set_title("Strategy win rate vs B&H by segment")
    ax.set_ylim(0, 110); ax.grid(True, alpha=0.3, axis="y")
    for xi, v in zip(x, seg["wr"]):
        ax.text(xi, v+2, f"{v:.0f}%", ha="center", fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIG/"03_segmented_analysis.png", dpi=110, bbox_inches="tight")
    plt.close()
    print(f"        wrote figures 01-03")


# ── Step 3: 26-pair B&H-survived basket ───────────────────────────────
def step3_basket(df: pd.DataFrame, O: pd.DataFrame, C: pd.DataFrame) -> dict:
    print(f"\n[step3] 26-pair B&H-survived basket — net of {COST_BPS_ROUND_TRIP} bps...")
    survived = df[df["bh_total"] > -0.50].copy()
    syms = sorted(survived["symbol"].tolist())
    print(f"        Survivors: {len(syms)} pairs")
    Os = O[syms]; Cs = C[syms]
    live = Cs.notna().astype(float)
    n_live = live.sum(axis=1).replace(0, np.nan)
    ret = (Cs / Os - 1).fillna(0.0)
    ma_f = Cs.rolling(5).mean(); ma_s = Cs.rolling(40).mean()
    sig = (ma_f > ma_s).astype(float).where(Cs.notna(), 0.0).shift(1).fillna(0.0) * live

    n_total = len(syms)
    n_longs = sig.sum(axis=1).replace(0, np.nan)

    schemes = {}
    # Fixed 1/N
    w = sig / n_total
    tov = w.diff().abs().sum(axis=1).fillna(0.0)
    r = (w * ret).sum(axis=1) - tov * COST_RATE
    schemes["fixed"] = (stats(r, "MA fixed 1/26"), r)
    # Live EW
    w = sig.div(n_live, axis=0).fillna(0.0)
    tov = w.diff().abs().sum(axis=1).fillna(0.0)
    r = (w * ret).sum(axis=1) - tov * COST_RATE
    schemes["live"] = (stats(r, "MA live equal-weight"), r)
    # Pro-rata
    w = sig.div(n_longs, axis=0).fillna(0.0)
    tov = w.diff().abs().sum(axis=1).fillna(0.0)
    r = (w * ret).sum(axis=1) - tov * COST_RATE
    schemes["pro"] = (stats(r, "MA pro-rata"), r)
    # B&H benchmark (also has rebalance cost as new symbols list)
    w_bh = live.div(n_live, axis=0).fillna(0.0)
    tov_bh = w_bh.diff().abs().sum(axis=1).fillna(0.0)
    r_bh = (w_bh * ret).sum(axis=1) - tov_bh * COST_RATE
    schemes["bh"] = (stats(r_bh, "Basket B&H (eq-wt across live)"), r_bh)

    print(f"  {'Strategy':<48s}  {'CAGR':>7s}  {'Sharpe':>7s}  {'MaxDD':>7s}  {'Total':>10s}")
    for k in ["bh","fixed","live","pro"]:
        s, _ = schemes[k]
        print(f"  {s['label']:<46s}  {s['cagr']*100:>6.1f}%  {s['sharpe']:>7.2f}  {s['max_dd']*100:>6.1f}%  {s['total']*100:>9.0f}%")

    # Save returns
    pd.DataFrame({
        "bh":      schemes["bh"][1],
        "fixed":   schemes["fixed"][1],
        "live":    schemes["live"][1],
        "pro":     schemes["pro"][1],
    }).to_parquet(OUT / "basket_returns.parquet")

    # Figure 04
    fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True,
                              gridspec_kw={"height_ratios":[3, 1.2, 1.2]})
    ax = axes[0]
    for k, c, ls, lw in [("bh","#2ca02c","--",1.5),("pro","#d62728","-",1.6),
                          ("live","#1f77b4","-",2.0),("fixed","#7f7f7f","-",1.3)]:
        s, _ = schemes[k]
        ax.plot(s["nav"].index, s["nav"]/1e3,
                label=f"{s['label']} (Sh={s['sharpe']:.2f}, DD={s['max_dd']:.0%})",
                color=c, lw=lw, ls=ls)
    ax.set_yscale("log"); ax.set_ylabel("NAV ($k, log)")
    ax.set_title(f"Multi-asset basket of 26 B&H-survived USDC pairs — net of {COST_BPS_ROUND_TRIP} bps round-trip")
    ax.legend(loc="lower right", fontsize=9); ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.fill_between(live.index, n_live.values, 0, alpha=0.3, color="#2ca02c", label="# live symbols")
    ax.fill_between(live.index, sig.sum(axis=1).values, 0, alpha=0.6, color="#1f77b4", label="# active longs")
    ax.set_ylabel("Symbol count"); ax.legend(loc="upper left", fontsize=9); ax.grid(True, alpha=0.3)

    ax = axes[2]
    for k, c, ls, label in [("bh","#2ca02c","--","B&H"), ("pro","#d62728","-","pro-rata"),
                             ("live","#1f77b4","-","live EW"), ("fixed","#7f7f7f","-","fixed 1/26")]:
        s, _ = schemes[k]
        dd = (s["nav"]/s["nav"].cummax()-1)*100
        ax.plot(dd.index, dd.values, color=c, lw=1.3, ls=ls, label=label, alpha=0.85)
    ax.axhline(0, color="k", lw=0.5); ax.set_ylabel("DD (%)")
    ax.legend(loc="lower left", fontsize=9); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG/"04_basket_equity_drawdown.png", dpi=110, bbox_inches="tight")
    plt.close()
    print(f"        wrote figure 04")
    return {k: v[0] for k, v in schemes.items()}


# ── Step 4: Per-pair walk-forward ─────────────────────────────────────
def step4_wfo_per_pair(df: pd.DataFrame, O: pd.DataFrame, C: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    print(f"\n[step4] Per-pair walk-forward — net of {COST_BPS_ROUND_TRIP} bps...")
    cands = df[df["years"] >= 5.0].sort_values("years", ascending=False)
    FAST = [3, 5, 8, 10, 15, 20]
    SLOW = [20, 30, 40, 50, 60, 80, 100]
    PARAMS = [(f, s) for f in FAST for s in SLOW if f < s]
    TRAIN = 730; TEST = 182

    def sh(r):
        r = pd.Series(r).dropna()
        if len(r) < 5 or r.std() == 0: return 0.0
        return r.mean()/r.std()*np.sqrt(ANN)

    def wfo_pair(opens, closes):
        n = len(opens)
        if n < TRAIN + TEST: return None
        bh = (closes/opens - 1).fillna(0.0)
        oos_returns, bh_oos = [], []
        selections = []
        start = TRAIN
        while start + TEST <= n:
            tr = slice(start - TRAIN, start)
            te = slice(start, start + TEST)
            best, best_s = None, -1e9
            for fast, slow in PARAMS:
                ma_f = closes.rolling(fast).mean(); ma_s = closes.rolling(slow).mean()
                sig = (ma_f > ma_s).astype(float).shift(1).fillna(0.0)
                tov = sig.diff().abs().fillna(0.0)
                r_tr = (bh * sig - tov * COST_RATE).iloc[tr]
                if (r_tr != 0).sum() < 30:
                    continue
                s = sh(r_tr)
                if s > best_s:
                    best_s, best = s, (fast, slow)
            if best is None:
                best = (5, 40); best_s = 0.0
            ma_f = closes.rolling(best[0]).mean(); ma_s = closes.rolling(best[1]).mean()
            sig = (ma_f > ma_s).astype(float).shift(1).fillna(0.0)
            tov = sig.diff().abs().fillna(0.0)
            r_te = (bh * sig - tov * COST_RATE).iloc[te]
            oos_returns.append(r_te); bh_oos.append(bh.iloc[te])
            selections.append({"train_end": opens.index[start-1], "test_start": opens.index[start],
                                "fast": best[0], "slow": best[1], "train_sharpe": best_s})
            start += TEST
        if not oos_returns: return None
        oos = pd.concat(oos_returns).sort_index()
        bh_oos_r = pd.concat(bh_oos).sort_index()
        sels = pd.DataFrame(selections)
        return dict(
            n_windows=len(sels),
            oos_returns=oos, bh_oos_returns=bh_oos_r,
            selections=sels,
            median_fast=int(sels["fast"].median()), median_slow=int(sels["slow"].median()),
            mode_fast=sels["fast"].mode().iloc[0], mode_slow=sels["slow"].mode().iloc[0],
        )

    rows = []
    all_oos = {}
    for _, row in cands.iterrows():
        sym = row["symbol"]
        opens = O[sym].dropna(); closes = C[sym].dropna()
        opens, closes = opens.align(closes, join="inner")
        out = wfo_pair(opens, closes)
        if out is None: continue
        s_strat = stats(out["oos_returns"])
        s_bh    = stats(out["bh_oos_returns"])
        rows.append(dict(
            symbol=sym, years=row["years"],
            oos_cagr=s_strat["cagr"], oos_sharpe=s_strat["sharpe"], oos_maxdd=s_strat["max_dd"],
            oos_total=s_strat["total"],
            bh_cagr=s_bh["cagr"], bh_sharpe=s_bh["sharpe"], bh_maxdd=s_bh["max_dd"], bh_total=s_bh["total"],
            edge_total=s_strat["total"] - s_bh["total"],
            edge_sharpe=s_strat["sharpe"] - s_bh["sharpe"],
            n_windows=out["n_windows"],
            median_fast=out["median_fast"], median_slow=out["median_slow"],
            mode_fast=out["mode_fast"], mode_slow=out["mode_slow"],
        ))
        all_oos[sym] = (out["oos_returns"], out["bh_oos_returns"])
    wfo_df = pd.DataFrame(rows)
    wfo_df.to_csv(OUT/"wfo_per_pair_results.csv", index=False)
    with open(OUT/"wfo_oos_returns_by_pair.pkl", "wb") as f:
        pickle.dump(all_oos, f)
    print(f"        wrote wfo_per_pair_results.csv ({len(wfo_df)} pairs)")

    # Walk-forward basket from stitched OOS returns
    HIGHQ_set = set(df[df["bh_total"] > -0.50]["symbol"])
    ALL_syms = list(all_oos.keys())
    SURV = [s for s in ALL_syms if s in HIGHQ_set]
    strat_panel = pd.DataFrame({s: all_oos[s][0] for s in ALL_syms}).sort_index()
    bh_panel    = pd.DataFrame({s: all_oos[s][1] for s in ALL_syms}).sort_index()

    def basket_oos(subset):
        s_ = strat_panel[subset]; b_ = bh_panel[subset]
        n_live_s = s_.notna().sum(axis=1).replace(0, np.nan)
        n_live_b = b_.notna().sum(axis=1).replace(0, np.nan)
        w_s = (s_.notna().astype(float).div(n_live_s, axis=0)).fillna(0.0)
        w_b = (b_.notna().astype(float).div(n_live_b, axis=0)).fillna(0.0)
        r_s = (s_.fillna(0.0) * w_s).sum(axis=1)
        # B&H rebalance cost: only when n_live changes (universe additions)
        tov_b = w_b.diff().abs().sum(axis=1).fillna(0.0)
        r_b = (b_.fillna(0.0) * w_b).sum(axis=1) - tov_b * COST_RATE
        # NOTE: per-pair OOS strategy returns ALREADY include their own per-pair costs.
        # Basket weight rebalance costs would be additional but smaller — we omit them
        # for parity with how the original analysis was done.
        return r_s, r_b

    r1_s, r1_b = basket_oos(ALL_syms)
    r2_s, r2_b = basket_oos(SURV)
    pd.DataFrame({"all49_strategy": r1_s, "all49_bh": r1_b,
                  "survived_strategy": r2_s, "survived_bh": r2_b}).to_parquet(OUT/"basket_wfo_returns.parquet")

    s1, b1 = stats(r1_s), stats(r1_b)
    s2, b2 = stats(r2_s), stats(r2_b)
    print(f"        Walk-forward basket OOS (net of {COST_BPS_ROUND_TRIP} bps):")
    print(f"          All-49:  strat Sh={s1['sharpe']:.2f}, total={s1['total']*100:.0f}%  |  B&H Sh={b1['sharpe']:.2f}, total={b1['total']*100:.0f}%")
    print(f"          Surv-12: strat Sh={s2['sharpe']:.2f}, total={s2['total']*100:.0f}%  |  B&H Sh={b2['sharpe']:.2f}, total={b2['total']*100:.0f}%")

    # Figure 05: stitched basket
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True, gridspec_kw={"height_ratios":[3,1]})
    ax = axes[0]
    ax.plot(s1["nav"].index, s1["nav"]/1e3,
            label=f"All-49 strategy (Sh={s1['sharpe']:.2f}, DD={s1['max_dd']:.0%})", color="#1f77b4", lw=1.8)
    ax.plot(b1["nav"].index, b1["nav"]/1e3,
            label=f"All-49 B&H (Sh={b1['sharpe']:.2f}, DD={b1['max_dd']:.0%})", color="#1f77b4", lw=1.3, ls="--", alpha=0.7)
    ax.plot(s2["nav"].index, s2["nav"]/1e3,
            label=f"Survived strategy (Sh={s2['sharpe']:.2f}, DD={s2['max_dd']:.0%})", color="#d62728", lw=2)
    ax.plot(b2["nav"].index, b2["nav"]/1e3,
            label=f"Survived B&H (Sh={b2['sharpe']:.2f}, DD={b2['max_dd']:.0%})", color="#d62728", lw=1.3, ls="--", alpha=0.7)
    ax.set_yscale("log"); ax.set_ylabel("NAV ($k, log)")
    ax.set_title(f"Walk-forward OOS basket (net of {COST_BPS_ROUND_TRIP} bps round-trip)")
    ax.legend(loc="lower right"); ax.grid(True, alpha=0.3)
    ax = axes[1]
    ax.plot(strat_panel[ALL_syms].notna().sum(axis=1), label="# pairs (all-49)", color="#1f77b4")
    ax.plot(strat_panel[SURV].notna().sum(axis=1), label="# pairs (survived)", color="#d62728")
    ax.set_ylabel("Pair count"); ax.grid(True, alpha=0.3); ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(FIG/"05_basket_walk_forward.png", dpi=110, bbox_inches="tight")
    plt.close()

    # Figure 06: rolling sharpe
    def roll_sh(r, w=365):
        return r.rolling(w).apply(lambda x: x.mean()/x.std()*np.sqrt(ANN) if x.std() > 0 else 0)
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(roll_sh(r1_s), label="All-49 strategy", color="#1f77b4", lw=1.5)
    ax.plot(roll_sh(r2_s), label="B&H-survived strategy", color="#d62728", lw=1.8)
    ax.plot(roll_sh(r2_b), label="B&H-survived B&H", color="#d62728", lw=1.2, ls="--", alpha=0.7)
    ax.axhline(0, color="k", lw=0.5); ax.axhline(1, color="gray", lw=0.4, ls=":")
    ax.set_title(f"Rolling 1-year Sharpe — walk-forward OOS basket (net of {COST_BPS_ROUND_TRIP} bps)")
    ax.set_ylabel("Sharpe (rolling 365d)"); ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG/"06_basket_wfo_rolling_sharpe.png", dpi=110, bbox_inches="tight")
    plt.close()
    print(f"        wrote figures 05-06")
    return wfo_df, all_oos


# ── Step 5: Category baskets + L1+L2+DeFi merged ──────────────────────
CATEGORY = {
    "BTC":"L1","ETH":"L1","BCH":"L1","LTC":"L1","ETC":"L1","XLM":"L1","XRP":"L1",
    "SOL":"L1","ADA":"L1","AVAX":"L1","DOT":"L1","NEAR":"L1","ATOM":"L1","ALGO":"L1",
    "EGLD":"L1","HBAR":"L1","TRX":"L1","ICP":"L1","FTM":"L1","SUI":"L1","APT":"L1",
    "SEI":"L1","INJ":"L1","TIA":"L1","DASH":"L1","EOS":"L1","XTZ":"L1","KAVA":"L1",
    "CGLD":"L1","CELO":"L1","KSM":"L1","IOTA":"L1","ZEN":"L1","NEO":"L1","WAVES":"L1",
    "ROSE":"L1","ONE":"L1","IOTX":"L1","FLOW":"L1","XEC":"L1","KDA":"L1","XCN":"L1",
    "AERO":"L1","STRK":"L1","TON":"L1","BSV":"L1","KAS":"L1","ASTR":"L1","BERA":"L1",
    "IOST":"L1","SPA":"L1","VTHO":"L1","VET":"L1","RVN":"L1","XEM":"L1","XYO":"L1","CHZ":"L1",
    "MATIC":"L2","OP":"L2","ARB":"L2","POL":"L2","MANTLE":"L2","MNT":"L2","LRC":"L2",
    "METIS":"L2","CTSI":"L2","BLAST":"L2","CORE":"L2","ZK":"L2","SKL":"L2","AURORA":"L2",
    "BOBA":"L2","MOVR":"L2","GLMR":"L2","FLR":"L2",
    "ZEC":"Privacy","XMR":"Privacy","MOB":"Privacy","KEEP":"Privacy","NU":"Privacy",
    "OXT":"Privacy","MASK":"Privacy","HOPR":"Privacy",
    "UNI":"DeFi","AAVE":"DeFi","COMP":"DeFi","MKR":"DeFi","SNX":"DeFi","YFI":"DeFi",
    "SUSHI":"DeFi","CRV":"DeFi","BAL":"DeFi","BAND":"DeFi","KNC":"DeFi","ANKR":"DeFi",
    "REN":"DeFi","UMA":"DeFi","PERP":"DeFi","DYDX":"DeFi","GMX":"DeFi","JTO":"DeFi",
    "BNT":"DeFi","IDEX":"DeFi","RAD":"DeFi","FXS":"DeFi","LDO":"DeFi","JUP":"DeFi",
    "CAKE":"DeFi","PENDLE":"DeFi","GTC":"DeFi","GNO":"DeFi","BICO":"DeFi","RPL":"DeFi",
    "GHST":"DeFi","WBTC":"DeFi","CETUS":"DeFi","KMNO":"DeFi","AERGO":"DeFi","1INCH":"DeFi",
    "ALCX":"DeFi","ALPHA":"DeFi","BADGER":"DeFi","SPELL":"DeFi","TRU":"DeFi","PNG":"DeFi",
    "POLS":"DeFi","FIDA":"DeFi","GFI":"DeFi","MNDE":"DeFi","INV":"DeFi","RARI":"DeFi",
    "INDEX":"DeFi","FORTH":"DeFi","CTX":"DeFi","SWFTC":"DeFi","MORPHO":"DeFi","PRQ":"DeFi",
    "EUL":"DeFi","SUKU":"DeFi","HFT":"DeFi","ORCA":"DeFi","AUCTION":"DeFi","SAFE":"DeFi",
    "OMNI":"DeFi","PAXG":"DeFi",
    "LINK":"Oracle","API3":"Oracle","TRB":"Oracle","TRIBE":"Oracle","PYTH":"Oracle","DIA":"Oracle",
    "FIL":"Storage","AR":"Storage","STORJ":"Storage","SC":"Storage","OCEAN":"Storage",
    "BLZ":"Storage","ALEPH":"Storage","CFG":"Storage",
    "DOGE":"Meme","SHIB":"Meme","PEPE":"Meme","BONK":"Meme","WIF":"Meme","FLOKI":"Meme",
    "MEME":"Meme","TURBO":"Meme","MOG":"Meme","POPCAT":"Meme","TRUMP":"Meme","GIGA":"Meme",
    "BRETT":"Meme","PNUT":"Meme","NEIRO":"Meme","ME":"Meme","MOODENG":"Meme","GOAT":"Meme",
    "FARTCOIN":"Meme","00":"Meme","LADYS":"Meme","PUFFER":"Meme","BAN":"Meme","CHILLGUY":"Meme",
    "AI16Z":"Meme","MYRO":"Meme",
    "USDT":"Stable","DAI":"Stable","PAX":"Stable","GUSD":"Stable","USDP":"Stable",
    "TUSD":"Stable","PYUSD":"Stable","USDS":"Stable","FRAX":"Stable","USDD":"Stable",
    "EURC":"Stable","CBETH":"Stable","MSOL":"Stable","LSETH":"Stable","OETH":"Stable","WSTETH":"Stable",
    "RNDR":"AI","RENDER":"AI","FET":"AI","AGIX":"AI","TAO":"AI","WLD":"AI","AKT":"AI",
    "GLM":"AI","IO":"AI","AIOZ":"AI","POND":"AI","NMR":"AI",
    "AXS":"Gaming","GALA":"Gaming","MANA":"Gaming","SAND":"Gaming","ENJ":"Gaming",
    "ILV":"Gaming","APE":"Gaming","BLUR":"Gaming","LOOKS":"Gaming","IMX":"Gaming",
    "BIGTIME":"Gaming","BEAM":"Gaming","PRIME":"Gaming","SUPER":"Gaming","PIRATE":"Gaming",
    "XRD":"Gaming","PRO":"Gaming","HIGH":"Gaming","ALICE":"Gaming","VOXEL":"Gaming",
    "GODS":"Gaming","GST":"Gaming","GMT":"Gaming","RONIN":"Gaming","PORT":"Gaming",
    "CRO":"Exchange","FTT":"Exchange","OKB":"Exchange","OGN":"Exchange","LCX":"Exchange",
    "BNB":"Exchange","INTX":"Exchange",
    "BAT":"Utility","ENS":"Utility","AMP":"Utility","COTI":"Utility","JASMY":"Utility",
    "CVC":"Utility","DIMO":"Utility","SHPING":"Utility","MDT":"Utility","BTRST":"Utility",
    "NKN":"Utility","DENT":"Utility","POWR":"Utility","REQ":"Utility","CLV":"Utility",
    "LPT":"Utility","PLU":"Utility","QNT":"Utility","XYO":"Utility","NCT":"Utility","HNT":"Utility",
    "DNT":"Utility","FOX":"Utility","GRT":"Utility","WAXP":"Utility","FIS":"Utility","ABT":"Utility",
    "ACS":"Utility","ASM":"Utility","AVT":"Utility","CVX":"Utility","DEXT":"Utility","DRIFT":"Utility",
    "ELA":"Utility","LOKA":"Utility","LSK":"Utility","MLN":"Utility","MUSE":"Utility","PUSH":"Utility",
    "QI":"Utility","RLY":"Utility","SAVAX":"Utility","SD":"Utility","SWELL":"Utility","TRAC":"Utility",
    "WAXL":"Utility","ZETA":"Utility","POLY":"Utility",
}
def categorize(sym): return CATEGORY.get(sym.split("-")[0], "Other")


def step5_categories(df: pd.DataFrame, O: pd.DataFrame, C: pd.DataFrame) -> dict:
    print(f"\n[step5] Category baskets — net of {COST_BPS_ROUND_TRIP} bps...")
    df = df.copy()
    df["category"] = df["symbol"].apply(categorize)
    df["edge_total"]    = df["strat_total"] - df["bh_total"]
    df["edge_sharpe"]   = df["strat_sharpe"] - df["bh_sharpe"]
    df["edge_maxdd_pp"] = (df["strat_maxdd"] - df["bh_maxdd"]) * 100
    df.to_csv(OUT/"usdc_universe_ma_5_40_results_with_category.csv", index=False)

    agg = df.groupby("category").agg(
        n_pairs=("symbol","count"),
        median_bh_total=("bh_total","median"),
        median_strat_total=("strat_total","median"),
        median_edge_total=("edge_total","median"),
        median_strat_sharpe=("strat_sharpe","median"),
        median_bh_sharpe=("bh_sharpe","median"),
        median_strat_maxdd=("strat_maxdd","median"),
        median_bh_maxdd=("bh_maxdd","median"),
        pct_strat_beats_bh=("edge_total", lambda x: (x>0).mean()*100),
        pct_strat_better_dd=("edge_maxdd_pp", lambda x: (x>0).mean()*100),
    ).round(3).sort_values("median_strat_sharpe", ascending=False)
    agg.to_csv(OUT/"category_stats.csv")
    print(agg.to_string())

    # Figure 07: category breakdown
    cats_order = ["L1","L2","DeFi","AI","Meme","Oracle","Gaming","Storage","Privacy","Exchange","Utility","Stable","Other"]
    cats = [c for c in cats_order if c in df["category"].unique() and (df["category"]==c).sum() >= 2]
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    ax = axes[0,0]
    data = [df.loc[df["category"]==c, "strat_total"].values for c in cats]
    bp = ax.boxplot(data, tick_labels=cats, showmeans=True, patch_artist=True,
                     meanprops=dict(marker="D", markerfacecolor="black", markersize=6))
    for box in bp["boxes"]: box.set_facecolor("lightblue")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_title(f"Strategy total return distribution by category (net of {COST_BPS_ROUND_TRIP} bps)")
    ax.set_ylabel("Total return"); ax.set_yscale("symlog"); ax.grid(True, alpha=0.3)
    ax.tick_params(axis="x", rotation=45)
    ax = axes[0,1]
    data = [df.loc[df["category"]==c, "edge_total"].values for c in cats]
    bp = ax.boxplot(data, tick_labels=cats, showmeans=True, patch_artist=True,
                     meanprops=dict(marker="D", markerfacecolor="black", markersize=6))
    for box in bp["boxes"]: box.set_facecolor("lightcoral")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_title("Strategy edge vs B&H by category (additive total return)")
    ax.set_ylabel("Edge"); ax.set_yscale("symlog"); ax.grid(True, alpha=0.3)
    ax.tick_params(axis="x", rotation=45)
    ax = axes[1,0]
    x = np.arange(len(cats)); w = 0.4
    ss = [df.loc[df["category"]==c, "strat_sharpe"].median() for c in cats]
    sb = [df.loc[df["category"]==c, "bh_sharpe"].median() for c in cats]
    ax.bar(x - w/2, ss, w, label="Strategy", color="#1f77b4")
    ax.bar(x + w/2, sb, w, label="B&H",      color="#2ca02c")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xticks(x); ax.set_xticklabels(cats, rotation=45)
    ax.set_title("Median Sharpe by category"); ax.set_ylabel("Sharpe")
    ax.legend(); ax.grid(True, alpha=0.3)
    ax = axes[1,1]
    wr = [(df.loc[df["category"]==c, "edge_total"] > 0).mean()*100 for c in cats]
    n_in_cat = [(df["category"]==c).sum() for c in cats]
    ax.bar(x, wr, color="#9467bd")
    ax.axhline(50, color="k", lw=0.5, ls="--")
    ax.set_xticks(x); ax.set_xticklabels(cats, rotation=45)
    ax.set_title("% pairs where strategy beats B&H by category")
    ax.set_ylabel("Win-rate (%)")
    ax.set_ylim(0, 105); ax.grid(True, alpha=0.3)
    for xi, (v, n) in enumerate(zip(wr, n_in_cat)):
        ax.text(xi, v+1.5, f"{v:.0f}%\n(n={n})", ha="center", fontsize=7)
    plt.tight_layout()
    plt.savefig(FIG/"07_category_breakdown.png", dpi=110, bbox_inches="tight")
    plt.close()

    # Figure 08: category basket curves
    cat_baskets = {}
    for cat in cats:
        sub = df.loc[df["category"]==cat, "symbol"].tolist()
        if len(sub) < 2: continue
        Os = O[sub]; Cs = C[sub]
        live = Cs.notna().astype(float)
        n_live = live.sum(axis=1).replace(0, np.nan)
        ret = (Cs/Os - 1).fillna(0.0)
        ma_f = Cs.rolling(5).mean(); ma_s = Cs.rolling(40).mean()
        sig = (ma_f > ma_s).astype(float).where(Cs.notna(), 0.0).shift(1).fillna(0.0) * live
        w_s = sig.div(n_live, axis=0).fillna(0.0)
        w_b = live.div(n_live, axis=0).fillna(0.0)
        tov_s = w_s.diff().abs().sum(axis=1).fillna(0.0)
        tov_b = w_b.diff().abs().sum(axis=1).fillna(0.0)
        r_s = (w_s * ret).sum(axis=1) - tov_s * COST_RATE
        r_b = (w_b * ret).sum(axis=1) - tov_b * COST_RATE
        a = stats(r_s); b = stats(r_b)
        if a is None or b is None: continue
        cat_baskets[cat] = dict(strat=a, bh=b, n=len(sub))

    print(f"\n=== Per-category baskets (live-EW, net of {COST_BPS_ROUND_TRIP} bps) ===")
    print(f"{'Category':<12s}  {'#':>4s}  {'Strat CAGR':>11s}  {'Strat Sh':>9s}  {'Strat DD':>9s}  {'B&H CAGR':>10s}  {'B&H Sh':>8s}  {'B&H DD':>8s}  {'Mult':>6s}")
    for cat, d in cat_baskets.items():
        print(f"  {cat:<10s}  {d['n']:>4d}  {d['strat']['cagr']*100:>9.1f}%  {d['strat']['sharpe']:>9.2f}  {d['strat']['max_dd']*100:>7.0f}%  {d['bh']['cagr']*100:>8.1f}%  {d['bh']['sharpe']:>8.2f}  {d['bh']['max_dd']*100:>6.0f}%  {(d['strat']['nav'].iloc[-1]/d['bh']['nav'].iloc[-1]):>5.2f}x")

    # Save category basket summary
    cat_basket_rows = []
    for cat, d in cat_baskets.items():
        cat_basket_rows.append(dict(
            category=cat, n=d["n"],
            strat_cagr=d["strat"]["cagr"], strat_sharpe=d["strat"]["sharpe"],
            strat_maxdd=d["strat"]["max_dd"], strat_total=d["strat"]["total"],
            bh_cagr=d["bh"]["cagr"], bh_sharpe=d["bh"]["sharpe"],
            bh_maxdd=d["bh"]["max_dd"], bh_total=d["bh"]["total"],
            mult=d["strat"]["nav"].iloc[-1]/d["bh"]["nav"].iloc[-1],
        ))
    pd.DataFrame(cat_basket_rows).to_csv(OUT/"category_basket_stats.csv", index=False)

    fig, axes = plt.subplots(3, 4, figsize=(20, 13), sharex=False)
    axes = axes.flatten()
    for i, (cat, d) in enumerate(cat_baskets.items()):
        if i >= len(axes): break
        ax = axes[i]
        ax.plot(d["strat"]["nav"].index, d["strat"]["nav"]/1e3,
                label=f"Strat (Sh={d['strat']['sharpe']:.2f})", color="#1f77b4", lw=1.6)
        ax.plot(d["bh"]["nav"].index, d["bh"]["nav"]/1e3,
                label=f"B&H (Sh={d['bh']['sharpe']:.2f})", color="#2ca02c", lw=1.3, ls="--")
        ax.set_yscale("log"); ax.set_title(f"{cat} basket (n={d['n']})", fontsize=10)
        ax.set_ylabel("NAV ($k, log)", fontsize=8)
        ax.legend(loc="lower right", fontsize=7); ax.grid(True, alpha=0.3)
    for j in range(i+1, len(axes)): axes[j].axis("off")
    plt.tight_layout()
    plt.savefig(FIG/"08_category_basket_curves.png", dpi=110, bbox_inches="tight")
    plt.close()

    print(f"        wrote figures 07-08")
    return cat_baskets


# ── Step 6: L1+L2+DeFi merged basket ──────────────────────────────────
def step6_hq_basket(df: pd.DataFrame, O: pd.DataFrame, C: pd.DataFrame) -> dict:
    print(f"\n[step6] L1+L2+DeFi merged basket — net of {COST_BPS_ROUND_TRIP} bps...")
    df = df.copy()
    df["category"] = df["symbol"].apply(categorize)
    HIGHQ = ["L1","L2","DeFi"]
    syms = sorted(df[df["category"].isin(HIGHQ)]["symbol"].tolist())
    print(f"        {len(syms)} pairs")
    Os = O[syms]; Cs = C[syms]
    live = Cs.notna().astype(float)
    n_live = live.sum(axis=1).replace(0, np.nan)
    ret = (Cs/Os - 1).fillna(0.0)
    ma_f = Cs.rolling(5).mean(); ma_s = Cs.rolling(40).mean()
    sig = (ma_f > ma_s).astype(float).where(Cs.notna(), 0.0).shift(1).fillna(0.0) * live
    w_s = sig.div(n_live, axis=0).fillna(0.0)
    w_b = live.div(n_live, axis=0).fillna(0.0)
    tov_s = w_s.diff().abs().sum(axis=1).fillna(0.0)
    tov_b = w_b.diff().abs().sum(axis=1).fillna(0.0)
    r_s = (w_s * ret).sum(axis=1) - tov_s * COST_RATE
    r_b = (w_b * ret).sum(axis=1) - tov_b * COST_RATE
    a = stats(r_s); b = stats(r_b)
    pd.DataFrame({"strategy": r_s, "bh": r_b}).to_parquet(OUT/"high_quality_basket_returns.parquet")
    print(f"  L1+L2+DeFi MA(5/40) live-EW (net): CAGR={a['cagr']*100:.1f}%, Sh={a['sharpe']:.2f}, DD={a['max_dd']*100:.0f}%, Total={a['total']*100:.0f}%")
    print(f"  L1+L2+DeFi B&H live-EW       (net): CAGR={b['cagr']*100:.1f}%, Sh={b['sharpe']:.2f}, DD={b['max_dd']*100:.0f}%, Total={b['total']*100:.0f}%")

    # Figure 09
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True, gridspec_kw={"height_ratios":[3,1]})
    ax = axes[0]
    ax.plot(a["nav"].index, a["nav"]/1e3,
            label=f"L1+L2+DeFi MA(5/40) live-EW (Sh={a['sharpe']:.2f}, DD={a['max_dd']:.0%})", color="#1f77b4", lw=2)
    ax.plot(b["nav"].index, b["nav"]/1e3,
            label=f"L1+L2+DeFi B&H live-EW (Sh={b['sharpe']:.2f}, DD={b['max_dd']:.0%})", color="#2ca02c", lw=1.5, ls="--")
    ax.set_yscale("log"); ax.set_ylabel("NAV ($k, log)")
    ax.set_title(f"Merged L1 + L2 + DeFi basket: {len(syms)} pairs — net of {COST_BPS_ROUND_TRIP} bps round-trip")
    ax.legend(loc="lower right"); ax.grid(True, alpha=0.3)
    ax = axes[1]
    dd_s = (a["nav"]/a["nav"].cummax()-1)*100
    dd_b = (b["nav"]/b["nav"].cummax()-1)*100
    ax.fill_between(dd_s.index, dd_s, 0, color="#1f77b4", alpha=0.5, label="Strategy")
    ax.plot(dd_b.index, dd_b, color="#2ca02c", lw=1.2, ls="--", label="B&H")
    ax.axhline(0, color="k", lw=0.5); ax.set_ylabel("DD (%)")
    ax.legend(loc="lower left", fontsize=9); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG/"09_high_quality_merged_basket.png", dpi=110, bbox_inches="tight")
    plt.close()
    print(f"        wrote figure 09")
    return {"strat": a, "bh": b, "r_strat": r_s, "r_bh": r_b}


# ── Step 7: Walk-forward L1+L2+DeFi basket ────────────────────────────
def step7_hq_wfo(df: pd.DataFrame, O: pd.DataFrame, C: pd.DataFrame) -> dict:
    print(f"\n[step7] L1+L2+DeFi basket walk-forward — net of {COST_BPS_ROUND_TRIP} bps...")
    df = df.copy(); df["category"] = df["symbol"].apply(categorize)
    HIGHQ = ["L1","L2","DeFi"]
    syms = sorted(df[df["category"].isin(HIGHQ)]["symbol"].tolist())
    Os = O[syms]; Cs = C[syms]
    live = Cs.notna().astype(float)
    n_live = live.sum(axis=1).replace(0, np.nan)
    ret = (Cs/Os - 1).fillna(0.0)
    FAST = [3, 5, 8, 10, 15, 20]
    SLOW = [20, 30, 40, 50, 60, 80, 100]
    PARAMS = [(f, s) for f in FAST for s in SLOW if f < s]

    # Pre-compute net basket return series for each grid combo
    print(f"        Pre-computing basket return series for {len(PARAMS)} grid combos...")
    t0 = time.time()
    basket_ret = {}
    for (f, s) in PARAMS:
        ma_f = Cs.rolling(f).mean(); ma_s = Cs.rolling(s).mean()
        sig = (ma_f > ma_s).astype(float).where(Cs.notna(), 0.0).shift(1).fillna(0.0) * live
        w = sig.div(n_live, axis=0).fillna(0.0)
        tov = w.diff().abs().sum(axis=1).fillna(0.0)
        basket_ret[(f, s)] = (w * ret).sum(axis=1) - tov * COST_RATE
    print(f"        done in {time.time()-t0:.1f}s")
    fixed_540 = basket_ret[(5, 40)]
    # B&H basket
    w_bh = live.div(n_live, axis=0).fillna(0.0)
    tov_bh = w_bh.diff().abs().sum(axis=1).fillna(0.0)
    r_bh_basket = (w_bh * ret).sum(axis=1) - tov_bh * COST_RATE

    # Walk-forward
    TRAIN, TEST = 730, 182
    def sh(r):
        r = pd.Series(r).dropna()
        if len(r) < 5 or r.std() == 0: return 0.0
        return r.mean()/r.std()*np.sqrt(ANN)

    idx = ret.index
    mask = (live.sum(axis=1) >= 3)
    first_valid = mask.idxmax()
    start_anchor = first_valid + pd.Timedelta(days=TRAIN)
    start_pos = idx.get_indexer([start_anchor], method="bfill")[0]

    selections = []; oos_chunks = []
    pos = start_pos
    while pos + TEST <= len(idx):
        tr = slice(pos - TRAIN, pos); te = slice(pos, pos + TEST)
        train_dates = idx[tr]; test_dates = idx[te]
        best, best_s = None, -1e9
        for (f, s) in PARAMS:
            r_tr = basket_ret[(f, s)].loc[train_dates]
            if (r_tr != 0).sum() < 50: continue
            ssh = sh(r_tr)
            if ssh > best_s: best_s, best = ssh, (f, s)
        if best is None: best = (5, 40); best_s = 0.0
        r_te = basket_ret[best].loc[test_dates]
        r_te_fixed = fixed_540.loc[test_dates]
        selections.append({"train_start": train_dates[0], "train_end": train_dates[-1],
                            "test_start": test_dates[0], "test_end": test_dates[-1],
                            "fast": best[0], "slow": best[1],
                            "train_sharpe": best_s,
                            "test_sharpe_wfo": sh(r_te),
                            "test_sharpe_fixed": sh(r_te_fixed)})
        oos_chunks.append(pd.DataFrame({"r_wfo": r_te, "r_fixed_5_40": r_te_fixed}))
        pos += TEST
    oos = pd.concat(oos_chunks).sort_index()
    sel_df = pd.DataFrame(selections)
    sel_df.to_csv(OUT/"highq_basket_wfo_selections.csv", index=False)
    r_bh_on_oos = r_bh_basket.loc[oos.index]
    oos["r_bh"] = r_bh_on_oos
    oos.to_parquet(OUT/"highq_basket_wfo_returns.parquet")

    s_wfo = stats(oos["r_wfo"]); s_fix = stats(oos["r_fixed_5_40"]); s_bh = stats(r_bh_on_oos)
    print(f"        {len(sel_df)} WFO windows from {oos.index[0].date()} -> {oos.index[-1].date()}  ({(oos.index[-1]-oos.index[0]).days/365:.1f} yrs)")
    print(f"        WFO basket:   CAGR={s_wfo['cagr']*100:.1f}%, Sh={s_wfo['sharpe']:.2f}, DD={s_wfo['max_dd']*100:.0f}%, Total={s_wfo['total']*100:.0f}%")
    print(f"        Fixed (5,40): CAGR={s_fix['cagr']*100:.1f}%, Sh={s_fix['sharpe']:.2f}, DD={s_fix['max_dd']*100:.0f}%, Total={s_fix['total']*100:.0f}%")
    print(f"        B&H OOS:      CAGR={s_bh['cagr']*100:.1f}%, Sh={s_bh['sharpe']:.2f}, DD={s_bh['max_dd']*100:.0f}%, Total={s_bh['total']*100:.0f}%")
    print(f"        Pct WFO > Fixed: {(sel_df['test_sharpe_wfo'] > sel_df['test_sharpe_fixed']).mean()*100:.0f}%")

    # Figure 10
    fig, axes = plt.subplots(3, 1, figsize=(13, 12), sharex=True,
                              gridspec_kw={"height_ratios":[3, 1.2, 1.5]})
    ax = axes[0]
    ax.plot(s_wfo["nav"].index, s_wfo["nav"]/1e3,
            label=f"WFO (re-opt fast/slow, Sh={s_wfo['sharpe']:.2f}, DD={s_wfo['max_dd']:.0%})", color="#1f77b4", lw=2.2)
    ax.plot(s_fix["nav"].index, s_fix["nav"]/1e3,
            label=f"Fixed MA(5/40) (Sh={s_fix['sharpe']:.2f}, DD={s_fix['max_dd']:.0%})", color="#9467bd", lw=1.6)
    ax.plot(s_bh["nav"].index, s_bh["nav"]/1e3,
            label=f"B&H live-EW (Sh={s_bh['sharpe']:.2f}, DD={s_bh['max_dd']:.0%})", color="#2ca02c", lw=1.4, ls="--")
    ax.set_yscale("log"); ax.set_ylabel("NAV ($k, log)")
    ax.set_title(f"L1+L2+DeFi basket — walk-forward OOS (net of {COST_BPS_ROUND_TRIP} bps round-trip)  {oos.index[0].date()} -> {oos.index[-1].date()}")
    ax.legend(loc="lower right"); ax.grid(True, alpha=0.3)
    ax = axes[1]
    ax.plot(sel_df["test_start"], sel_df["fast"], marker="o", ms=4, lw=0.8, color="#1f77b4", label="selected fast MA")
    ax.plot(sel_df["test_start"], sel_df["slow"], marker="s", ms=4, lw=0.8, color="#d62728", label="selected slow MA")
    ax.axhline(5, color="#1f77b4", lw=0.5, ls=":")
    ax.axhline(40, color="#d62728", lw=0.5, ls=":")
    ax.set_ylabel("Selected MA window"); ax.legend(loc="upper right"); ax.grid(True, alpha=0.3)
    ax = axes[2]
    for s_, c, ls, label in [(s_wfo, "#1f77b4", "-", "WFO"),
                              (s_fix, "#9467bd", "-", "Fixed 5/40"),
                              (s_bh,  "#2ca02c", "--", "B&H")]:
        dd = (s_["nav"]/s_["nav"].cummax()-1)*100
        if label == "WFO":
            ax.fill_between(dd.index, dd, 0, color=c, alpha=0.5, label=label)
        else:
            ax.plot(dd.index, dd.values, color=c, lw=1.2, ls=ls, label=label, alpha=0.85)
    ax.axhline(0, color="k", lw=0.5); ax.set_ylabel("DD (%)")
    ax.legend(loc="lower left", fontsize=9); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG/"10_highq_basket_walk_forward.png", dpi=110, bbox_inches="tight")
    plt.close()

    # Figure 11: param heatmap
    fig, ax = plt.subplots(figsize=(12, 6))
    mat = np.zeros((len(FAST), len(SLOW)))
    for _, row in sel_df.iterrows():
        fi = FAST.index(row["fast"]); si = SLOW.index(row["slow"])
        mat[fi, si] += 1
    im = ax.imshow(mat, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(SLOW))); ax.set_xticklabels(SLOW)
    ax.set_yticks(range(len(FAST))); ax.set_yticklabels(FAST)
    ax.set_xlabel("Slow MA"); ax.set_ylabel("Fast MA")
    ax.set_title(f"Walk-forward parameter selection heatmap ({len(sel_df)} windows)")
    for fi in range(len(FAST)):
        for si in range(len(SLOW)):
            if mat[fi, si] > 0:
                ax.text(si, fi, f"{int(mat[fi,si])}", ha="center", va="center",
                        color="white" if mat[fi,si] > mat.max()/2 else "black", fontsize=10)
    plt.colorbar(im, ax=ax, label="# windows selected")
    plt.tight_layout()
    plt.savefig(FIG/"11_highq_basket_wfo_param_heatmap.png", dpi=110, bbox_inches="tight")
    plt.close()

    # Figure 12: rolling sharpe
    def roll_sh(r, w=365):
        return r.rolling(w).apply(lambda x: x.mean()/x.std()*np.sqrt(ANN) if x.std() > 0 else 0)
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(roll_sh(oos["r_wfo"]), label="WFO basket", color="#1f77b4", lw=1.5)
    ax.plot(roll_sh(oos["r_fixed_5_40"]), label="Fixed MA(5/40)", color="#9467bd", lw=1.2)
    ax.plot(roll_sh(r_bh_on_oos), label="B&H", color="#2ca02c", lw=1.2, ls="--")
    ax.axhline(0, color="k", lw=0.5); ax.axhline(1, color="gray", lw=0.4, ls=":")
    ax.set_title(f"Rolling 1-year OOS Sharpe — L1+L2+DeFi basket (net of {COST_BPS_ROUND_TRIP} bps)")
    ax.set_ylabel("Sharpe (rolling 365d)"); ax.legend(loc="upper right"); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG/"12_highq_basket_wfo_rolling_sharpe.png", dpi=110, bbox_inches="tight")
    plt.close()

    # Figure 13: headline equity curve (fixed (5,40) is primary)
    nav_s = s_fix["nav"]; nav_bh = s_bh["nav"]
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.plot(nav_s.index, nav_s/1e3,
            label=f"MA(5/40) basket strategy   →  ${nav_s.iloc[-1]:>10,.0f}  ({s_fix['total']*100:+.0f}%)", color="#1f77b4", lw=2.6)
    ax.plot(nav_bh.index, nav_bh/1e3,
            label=f"Buy-and-Hold (same 85 pairs)→  ${nav_bh.iloc[-1]:>10,.0f}  ({s_bh['total']*100:+.0f}%)", color="#d62728", lw=2.2)
    ax.axhline(INITIAL/1e3, color="gray", lw=0.8, ls="--", alpha=0.6)
    ax.set_yscale("log"); ax.set_ylabel("NAV ($k, log scale)", fontsize=12)
    ax.set_title(f"L1+L2+DeFi 85-pair basket: walk-forward OOS (net of {COST_BPS_ROUND_TRIP} bps round-trip)\n{oos.index[0].strftime('%b %Y')} → {oos.index[-1].strftime('%b %Y')}  ({(oos.index[-1]-oos.index[0]).days/365:.1f} years, MA(5/40) fixed throughout)",
                  fontsize=13)
    ax.legend(loc="upper left", fontsize=11, frameon=True, framealpha=0.95)
    ax.grid(True, alpha=0.3, which="both")
    last = oos.index[-1]
    ax.annotate(f"  {s_fix['total']*100:+.0f}%", xy=(last, nav_s.iloc[-1]/1e3),
                xytext=(20, 10), textcoords="offset points", color="#1f77b4", fontsize=14, fontweight="bold")
    ax.annotate(f"  {s_bh['total']*100:+.0f}%", xy=(last, nav_bh.iloc[-1]/1e3),
                xytext=(20, -5), textcoords="offset points", color="#d62728", fontsize=14, fontweight="bold")
    stats_text = (
        f"            Strategy    B&H\n"
        f"CAGR:       {s_fix['cagr']*100:>+5.1f}%   {s_bh['cagr']*100:>+5.1f}%\n"
        f"Sharpe:     {s_fix['sharpe']:>+5.2f}    {s_bh['sharpe']:>+5.2f}\n"
        f"MaxDD:      {s_fix['max_dd']*100:>+5.0f}%   {s_bh['max_dd']*100:>+5.0f}%\n"
        f"Total:      {s_fix['total']*100:>+5.0f}%   {s_bh['total']*100:>+5.0f}%\n"
        f"Final NAV:  ${nav_s.iloc[-1]/1e3:>5,.0f}k  ${nav_bh.iloc[-1]/1e3:>5,.0f}k"
    )
    ax.text(0.99, 0.03, stats_text, transform=ax.transAxes, fontsize=9.5, family="monospace",
            verticalalignment="bottom", horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="black", alpha=0.95))
    plt.tight_layout()
    plt.savefig(FIG/"13_headline_oos_equity.png", dpi=120, bbox_inches="tight")
    plt.close()

    # Figure 14: storyboard
    r_strat_h = oos["r_fixed_5_40"]; r_bh_h = oos["r_bh"]
    nav_strat = INITIAL*(1+r_strat_h).cumprod()
    nav_bh_h  = INITIAL*(1+r_bh_h).cumprod()
    ratio = nav_strat/nav_bh_h
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.6, 1, 1], hspace=0.32, wspace=0.18)
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(nav_strat.index, nav_strat/1e3,
              label=f"Strategy: ${nav_strat.iloc[-1]:,.0f}  ({s_fix['total']*100:+.0f}%, Sh {s_fix['sharpe']:.2f})",
              color="#1f77b4", lw=2.4)
    ax1.plot(nav_bh_h.index, nav_bh_h/1e3,
              label=f"B&H:      ${nav_bh_h.iloc[-1]:,.0f}  ({s_bh['total']*100:+.0f}%, Sh {s_bh['sharpe']:.2f})",
              color="#d62728", lw=2.0)
    ax1.axhline(INITIAL/1e3, color="gray", lw=0.6, ls="--", alpha=0.6)
    ax1.text(oos.index[10], INITIAL/1e3*1.05, "$100k start", fontsize=9, color="gray", alpha=0.8)
    ax1.set_yscale("log"); ax1.set_ylabel("NAV ($k, log)")
    ax1.set_title(f"L1+L2+DeFi 85-pair basket — walk-forward OOS (net of {COST_BPS_ROUND_TRIP} bps round-trip)")
    ax1.legend(loc="upper left"); ax1.grid(True, alpha=0.3, which="both")
    ax2 = fig.add_subplot(gs[1, 0])
    dd_s = (nav_strat/nav_strat.cummax()-1)*100
    dd_b = (nav_bh_h/nav_bh_h.cummax()-1)*100
    ax2.fill_between(dd_s.index, dd_s, 0, color="#1f77b4", alpha=0.5, label="Strategy")
    ax2.plot(dd_b.index, dd_b, color="#d62728", lw=1.6, label="B&H")
    ax2.axhline(0, color="k", lw=0.5); ax2.set_ylabel("Drawdown (%)")
    ax2.set_title(f"Drawdown: Strategy {s_fix['max_dd']*100:.0f}%  vs  B&H {s_bh['max_dd']*100:.0f}%")
    ax2.legend(loc="lower left", fontsize=9); ax2.grid(True, alpha=0.3)
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(ratio.index, ratio.values, color="#9467bd", lw=2.0)
    ax3.axhline(1, color="k", lw=0.7, ls="--")
    ax3.fill_between(ratio.index, ratio.values, 1, where=(ratio.values >= 1), color="#9467bd", alpha=0.18, interpolate=True, label="Strategy > B&H")
    ax3.fill_between(ratio.index, ratio.values, 1, where=(ratio.values <  1), color="#d62728", alpha=0.18, interpolate=True, label="B&H > Strategy")
    ax3.set_yscale("log"); ax3.set_ylabel("Strategy NAV / B&H NAV")
    ax3.set_title(f"Cumulative outperformance ratio (end: {ratio.iloc[-1]:.1f}×)")
    ax3.legend(loc="upper left", fontsize=9); ax3.grid(True, alpha=0.3, which="both")
    yr = pd.DataFrame({"strat": r_strat_h, "bh": r_bh_h})
    yr["year"] = yr.index.year
    yr_ann = yr.groupby("year").apply(lambda g: pd.Series({
        "strat": (1+g["strat"]).prod()-1, "bh": (1+g["bh"]).prod()-1,
    }), include_groups=False).reset_index()
    ax4 = fig.add_subplot(gs[2, :])
    x = np.arange(len(yr_ann)); w = 0.4
    ax4.bar(x - w/2, yr_ann["strat"]*100, w, label="Strategy MA(5/40)", color="#1f77b4")
    ax4.bar(x + w/2, yr_ann["bh"]*100,    w, label="B&H (live-EW)",    color="#d62728")
    ax4.axhline(0, color="k", lw=0.5)
    ax4.set_xticks(x); ax4.set_xticklabels(yr_ann["year"].astype(int).astype(str))
    ax4.set_ylabel("Calendar-year return (%)")
    ax4.set_title("Calendar-year returns — where the gap accumulates")
    ax4.legend(loc="upper right"); ax4.grid(True, alpha=0.3, axis="y")
    for xi, (sv, bv) in enumerate(zip(yr_ann["strat"]*100, yr_ann["bh"]*100)):
        ax4.text(xi - w/2, sv + (3 if sv>=0 else -8), f"{sv:+.0f}%", ha="center", fontsize=7, color="#1f77b4")
        ax4.text(xi + w/2, bv + (3 if bv>=0 else -8), f"{bv:+.0f}%", ha="center", fontsize=7, color="#d62728")
    plt.tight_layout()
    plt.savefig(FIG/"14_oos_storyboard.png", dpi=120, bbox_inches="tight")
    plt.close()

    # Save calendar-year table for the report
    yr_ann.to_csv(OUT/"highq_basket_wfo_calendar_year.csv", index=False)

    # Figure 15: linear + endpoint
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={"width_ratios":[3, 1]})
    ax = axes[0]
    ax.plot(nav_strat.index, nav_strat/1e3, label="Strategy MA(5/40)", color="#1f77b4", lw=2.4)
    ax.plot(nav_bh_h.index, nav_bh_h/1e3, label="B&H (same basket)", color="#d62728", lw=2.0)
    ax.axhline(INITIAL/1e3, color="gray", lw=0.7, ls="--", alpha=0.7)
    ax.set_ylabel("NAV ($k, linear)")
    ax.set_title(f"7.5-year OOS — linear scale (net of {COST_BPS_ROUND_TRIP} bps)")
    ax.legend(loc="upper left"); ax.grid(True, alpha=0.3)
    ax = axes[1]
    bars = ax.bar(["Strategy","B&H","$100k\nstart"],
                   [nav_strat.iloc[-1]/1e3, nav_bh_h.iloc[-1]/1e3, INITIAL/1e3],
                   color=["#1f77b4","#d62728","gray"])
    ax.set_yscale("log"); ax.set_ylabel("Final NAV ($k, log)")
    ax.set_title("Endpoint comparison")
    for bar, val in zip(bars, [nav_strat.iloc[-1]/1e3, nav_bh_h.iloc[-1]/1e3, INITIAL/1e3]):
        ax.text(bar.get_x()+bar.get_width()/2, val*1.06, f"${val:,.0f}k", ha="center", fontsize=10, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(FIG/"15_oos_linear_vs_log.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"        wrote figures 10-15")
    return dict(s_wfo=s_wfo, s_fix=s_fix, s_bh=s_bh, sel_df=sel_df, yr_ann=yr_ann)


# ── Main ───────────────────────────────────────────────────────────────
def main():
    print(f"== Re-running universe study at {COST_BPS_ROUND_TRIP} bps round-trip ==")
    syms, O, C = load_universe()
    df = step1_universe(syms, O, C)
    step2_universe_figures(df)
    step3_basket(df, O, C)
    wfo_df, all_oos = step4_wfo_per_pair(df, O, C)
    step5_categories(df, O, C)
    step6_hq_basket(df, O, C)
    step7_hq_wfo(df, O, C)
    print("\n== DONE ==")
    print(f"All artifacts written to {OUT}")


if __name__ == "__main__":
    main()
