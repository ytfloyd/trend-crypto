"""Per-symbol feature computation for the gamma screener.

Reads the latest ``vol_surface_snaps`` row-group for each symbol, reconstructs
a ``VolSurface``, and extracts constant-maturity IV at the configured tenors.
Pulls ``bars_1d`` for realized-vol estimators. Produces one ``FeatureRow``
per symbol, ready for cross-sectional scoring.

Conventions
-----------
- RV is computed with ``volatility.estimators`` on daily bars (ann_factor=252).
- IV is extracted via ``VolSurface.iv(strike=forward, tte_years)`` so we get
  forward-ATM IV with variance-space interpolation across expiries.
- Spot is taken from the latest snapshot's underlying_price.
- All times are in years (tte = days / 365.25) for IV, but RV is trading-day
  annualised. We are consistent with what each source uses internally.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Optional

import duckdb
import numpy as np
import pandas as pd

from common.logging import get_logger
from data.options.snapshot import IBVolSurfaceCollector
from volatility import estimators as vol_est
from volatility.surface import VolSurface

from .config import GammaScreenerConfig

logger = get_logger("gamma_screener_signals")

TRADING_DAYS_PER_YEAR = 252.0


@dataclass(frozen=True)
class FeatureRow:
    """Raw per-symbol features, pre-scoring."""
    as_of_date: date
    symbol: str
    spot: float
    iv7: Optional[float]
    iv30: Optional[float]
    iv60: Optional[float]
    iv90: Optional[float]
    rv_cc10: Optional[float]
    rv_cc20: Optional[float]
    rv_yz20: Optional[float]
    rv_yz60: Optional[float]
    iv30_rv20_ratio: Optional[float]
    iv7_rv10_ratio: Optional[float]
    term_30_90: Optional[float]
    term_7_30: Optional[float]
    skew_25d_30: Optional[float]
    butterfly_25d_30: Optional[float]
    iv_rank_252: Optional[float]
    bid_ask_pct: Optional[float]
    stock_adv_usd: Optional[float]
    options_adv_usd: Optional[float]
    earnings_in_window: bool


def _load_bars(
    conn: duckdb.DuckDBPyConnection,
    symbol: str,
    table: str,
    lookback_days: int = 400,
) -> pd.DataFrame:
    """Load recent daily OHLCV bars for a symbol."""
    start_ts = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).date()
    df = conn.execute(f"""
        SELECT ts, open, high, low, close, volume
        FROM {table}
        WHERE symbol = ?
          AND ts >= ?
        ORDER BY ts
    """, [symbol, start_ts]).fetchdf()
    if df.empty:
        return df
    df["ts"] = pd.to_datetime(df["ts"])
    df = df.set_index("ts")
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = df[c].astype(float)
    return df


def _iv_at_tenor(surface: VolSurface, days: float) -> Optional[float]:
    """Forward-ATM IV at an arbitrary tenor in days."""
    if not surface.slices:
        return None
    tte = days / 365.25
    # Use the nearest slice's forward as the ATM strike; acceptable for
    # equities where dividend yield is small.
    ref_slice = surface.nearest_slice(tte)
    forward = ref_slice.forward
    try:
        iv = float(surface.iv(strike=forward, tte=tte))
    except Exception:  # noqa: BLE001
        return None
    if not np.isfinite(iv) or iv <= 0:
        return None
    return iv


def _skew_and_butterfly_at_30d(surface: VolSurface) -> tuple[Optional[float], Optional[float]]:
    """25-delta skew and butterfly, interpolated at the 30d slice."""
    if not surface.slices:
        return None, None
    target_tte = 30.0 / 365.25
    s = surface.nearest_slice(target_tte)
    try:
        skew = float(s.skew_25d())
        bf = float(s.butterfly_25d())
    except Exception:  # noqa: BLE001
        return None, None
    skew = skew if np.isfinite(skew) else None
    bf = bf if np.isfinite(bf) else None
    return skew, bf


def _compute_rvs(bars: pd.DataFrame, windows: tuple[int, ...]) -> dict[str, Optional[float]]:
    """Compute a handful of realized-vol estimators on daily bars."""
    out: dict[str, Optional[float]] = {}
    if bars.empty or len(bars) < max(windows) + 5:
        for w in windows:
            out[f"rv_cc{w}"] = None
            out[f"rv_yz{w}"] = None
        return out

    close = bars["close"]
    for w in windows:
        cc = vol_est.close_to_close(close, window=w, ann_factor=TRADING_DAYS_PER_YEAR)
        yz = vol_est.yang_zhang(
            bars["open"], bars["high"], bars["low"], bars["close"],
            window=w, ann_factor=TRADING_DAYS_PER_YEAR,
        )
        cc_v = cc.dropna().iloc[-1] if not cc.dropna().empty else None
        yz_v = yz.dropna().iloc[-1] if not yz.dropna().empty else None
        out[f"rv_cc{w}"] = float(cc_v) if cc_v is not None else None
        out[f"rv_yz{w}"] = float(yz_v) if yz_v is not None else None
    return out


def _stock_adv_usd(bars: pd.DataFrame, window: int = 20) -> Optional[float]:
    """Rolling average dollar-volume (spot * share_volume) over last N days."""
    if bars.empty:
        return None
    tail = bars.tail(window)
    if tail.empty:
        return None
    dollar_vol = (tail["close"] * tail["volume"]).dropna()
    if dollar_vol.empty:
        return None
    return float(dollar_vol.mean())


def _bid_ask_and_options_adv(
    conn: duckdb.DuckDBPyConnection,
    symbol: str,
    snap_ts: datetime,
) -> tuple[Optional[float], Optional[float]]:
    """Derive ATM bid-ask % and options $ ADV from the latest snapshot."""
    # ATM bid-ask spread (nearest 30d, strike closest to forward)
    target_tte = 30.0 / 365.25
    rows = conn.execute("""
        SELECT strike, forward, mid_iv, bid_iv, ask_iv,
               mid_price, volume, underlying_price, tte_years
        FROM vol_surface_snaps
        WHERE underlying = ? AND snap_ts = ? AND "right" = 'C'
    """, [symbol, snap_ts]).fetchdf()
    if rows.empty:
        return None, None

    rows["abs_dt"] = (rows["tte_years"] - target_tte).abs()
    nearest_tte = rows.loc[rows["abs_dt"].idxmin(), "tte_years"]
    slice_df = rows[rows["tte_years"] == nearest_tte].copy()
    if slice_df.empty:
        return None, None
    slice_df["abs_dk"] = (slice_df["strike"] - slice_df["forward"]).abs()
    atm_row = slice_df.loc[slice_df["abs_dk"].idxmin()]

    bid_ask_pct: Optional[float] = None
    if (
        pd.notna(atm_row.get("bid_iv"))
        and pd.notna(atm_row.get("ask_iv"))
        and pd.notna(atm_row.get("mid_iv"))
        and float(atm_row["mid_iv"]) > 0
    ):
        bid_ask_pct = float(
            (atm_row["ask_iv"] - atm_row["bid_iv"]) / atm_row["mid_iv"]
        )

    # Options $ADV — rough proxy: sum(volume * mid_price * 100) over the snap
    vol_col = rows["volume"].fillna(0.0)
    px_col = rows["mid_price"].fillna(0.0)
    options_adv_usd = float((vol_col * px_col * 100.0).sum())
    if options_adv_usd <= 0:
        options_adv_usd = None  # type: ignore[assignment]

    return bid_ask_pct, options_adv_usd


def _iv_rank_252(
    conn: duckdb.DuckDBPyConnection,
    symbol: str,
    current_iv30: Optional[float],
    lookback_days: int,
    min_history: int,
) -> Optional[float]:
    """Historical IV-30 rank — percentile of today's IV30 over trailing window.

    Returns None if we don't yet have enough daily snapshots. This lets the
    score gracefully degrade during the bootstrap period.
    """
    if current_iv30 is None:
        return None
    start_ts = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    hist = conn.execute("""
        SELECT as_of_date, iv30
        FROM gamma_screener_daily
        WHERE symbol = ?
          AND as_of_date >= ?
          AND iv30 IS NOT NULL
    """, [symbol, start_ts.date()]).fetchdf()
    if len(hist) < min_history:
        return None
    series = hist["iv30"].dropna().astype(float)
    if series.empty:
        return None
    pct = float((series <= current_iv30).mean())
    return pct


def compute_features(
    symbols: list[str],
    cfg: GammaScreenerConfig,
    as_of_date: Optional[date] = None,
    earnings_symbols_in_window: Optional[set[str]] = None,
) -> list[FeatureRow]:
    """Compute FeatureRow for each symbol from DuckDB state.

    Assumes ``snapshot_universe()`` has already populated ``vol_surface_snaps``
    and that ``bars_1d`` has recent equity bars.
    """
    as_of_date = as_of_date or datetime.now(timezone.utc).date()
    earnings_symbols_in_window = earnings_symbols_in_window or set()

    conn = duckdb.connect(cfg.stocks_db_path, read_only=False)
    collector = IBVolSurfaceCollector.__new__(IBVolSurfaceCollector)
    collector._conn = conn
    collector._ib = None
    collector.db_path = cfg.stocks_db_path

    rows: list[FeatureRow] = []

    for symbol in symbols:
        try:
            surface = collector.load_surface(symbol)
        except Exception as exc:  # noqa: BLE001
            logger.debug("%s: no surface loadable (%s)", symbol, exc)
            continue

        if len(surface.slices) < cfg.min_slices_required:
            continue

        spot = float(surface.slices[0].underlying_price)
        if spot < cfg.min_spot_price:
            continue

        iv7 = _iv_at_tenor(surface, 7.0)
        iv30 = _iv_at_tenor(surface, 30.0)
        iv60 = _iv_at_tenor(surface, 60.0)
        iv90 = _iv_at_tenor(surface, 90.0)

        skew_30, bf_30 = _skew_and_butterfly_at_30d(surface)

        bars = _load_bars(conn, symbol, cfg.bars_table)
        rvs = _compute_rvs(bars, cfg.rv_windows)
        stock_adv = _stock_adv_usd(bars)

        bid_ask_pct, options_adv = _bid_ask_and_options_adv(
            conn, symbol, surface.snapshot_ts,
        )

        iv_rank = _iv_rank_252(
            conn, symbol, iv30,
            cfg.iv_rank_lookback_days, cfg.iv_rank_min_history,
        )

        rv_yz10 = rvs.get("rv_yz10")
        rv_yz20 = rvs.get("rv_yz20")
        iv30_rv20 = (iv30 / rv_yz20) if (iv30 and rv_yz20) else None
        iv7_rv10 = (iv7 / rv_yz10) if (iv7 and rv_yz10) else None
        term_30_90 = (iv30 - iv90) if iv30 and iv90 else None
        term_7_30 = (iv7 - iv30) if iv7 and iv30 else None

        rows.append(FeatureRow(
            as_of_date=as_of_date,
            symbol=symbol,
            spot=spot,
            iv7=iv7, iv30=iv30, iv60=iv60, iv90=iv90,
            rv_cc10=rvs.get("rv_cc10"),
            rv_cc20=rvs.get("rv_cc20"),
            rv_yz20=rvs.get("rv_yz20"),
            rv_yz60=rvs.get("rv_yz60"),
            iv30_rv20_ratio=iv30_rv20,
            iv7_rv10_ratio=iv7_rv10,
            term_30_90=term_30_90,
            term_7_30=term_7_30,
            skew_25d_30=skew_30,
            butterfly_25d_30=bf_30,
            iv_rank_252=iv_rank,
            bid_ask_pct=bid_ask_pct,
            stock_adv_usd=stock_adv,
            options_adv_usd=options_adv,
            earnings_in_window=symbol in earnings_symbols_in_window,
        ))

    conn.close()
    logger.info("Computed features for %d / %d symbols", len(rows), len(symbols))
    return rows
