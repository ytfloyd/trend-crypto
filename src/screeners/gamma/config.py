"""Configuration for the underpriced-gamma screener.

All tuning lives here. Thresholds, score weights, filter cutoffs, IB
connection params, and output paths. Defaults are tuned for S&P 100
end-of-day operation; tighten for production.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class GammaScreenerConfig:
    # ── Universe & DB ─────────────────────────────────────────────
    universe_name: str = "sp100"
    stocks_db_path: str = "../data/stocks_market.duckdb"
    bars_table: str = "bars_1d"

    # ── IB connection ─────────────────────────────────────────────
    ib_host: str = "127.0.0.1"
    ib_port: int = 7497  # TWS paper; 7496 live, 4001/4002 gateway
    ib_client_id: int = 20

    # ── Snapshot parameters ───────────────────────────────────────
    max_expiries: int = 8
    strike_range_pct: float = 0.25
    min_tte_days: int = 3
    max_tte_days: int = 120

    # ── Feature horizons (calendar days) ──────────────────────────
    iv_tenors_days: tuple[int, ...] = (7, 30, 60, 90)
    rv_windows: tuple[int, ...] = (10, 20, 60)  # trading days

    # ── Hard filters (rows failing these are excluded from ranking) ──
    min_spot_price: float = 5.0
    max_bid_ask_pct: float = 0.15       # ATM bid-ask as frac of mid
    min_iv: float = 0.05
    max_iv: float = 3.00
    min_stock_adv_usd: float = 10_000_000.0
    min_slices_required: int = 2         # need >=2 expiries to do term structure

    # ── Score weights (applied to cross-sectional z-scores) ───────
    weight_short: float = 1.0     # iv7_rv10 and iv7 z-score
    weight_thirty: float = 1.5    # iv30_rv20 — the workhorse
    weight_term: float = 0.5      # front-back spread
    weight_earnings_penalty: float = 0.75  # subtracted if earnings in TTE window

    # ── Outputs ───────────────────────────────────────────────────
    report_top_n: int = 25
    output_dir: str = "artifacts/gamma_screener"

    # ── IV-rank history (only used once enough days are captured) ──
    iv_rank_lookback_days: int = 252
    iv_rank_min_history: int = 63

    # ── Earnings ──────────────────────────────────────────────────
    earnings_lookahead_days: int = 45

    custom_tickers: tuple[str, ...] = field(default_factory=tuple)
