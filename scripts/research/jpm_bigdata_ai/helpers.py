"""
Shared helpers for JPM Big Data & AI Strategies research scripts.

Re-exports common infrastructure and adds ML-specific utilities
for feature engineering, walk-forward validation, and model evaluation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Re-export shared infrastructure
from scripts.research.common.data import (  # noqa: F401
    ANN_FACTOR,
    DEFAULT_DB,
    compute_btc_benchmark,
    filter_universe,
    load_daily_bars,
)
from scripts.research.common.backtest import (  # noqa: F401
    DEFAULT_COST_BPS,
    simple_backtest,
)
from scripts.research.common.metrics import (  # noqa: F401
    compute_metrics,
    format_metrics_table,
)
from scripts.research.common.risk_overlays import (  # noqa: F401
    apply_dd_control,
    apply_position_limit_dict,
    apply_position_limit_wide,
    apply_trailing_stop,
    apply_vol_targeting,
)

# ---------------------------------------------------------------------------
# Paper reference
# ---------------------------------------------------------------------------
PAPER_REF = (
    "Kolanovic & Krishnamachari (2017), "
    "'Big Data and AI Strategies: Machine Learning and "
    "Alternative Data Approach to Investing', JPMorgan"
)
PAPER_PATH = "/Users/russellfloyd/Dropbox/papers_studied/JPM-2017-MachineLearningInvestments.pdf"


# ---------------------------------------------------------------------------
# Feature engineering utilities (powered by TA-Lib)
# ---------------------------------------------------------------------------
RETURN_LOOKBACKS = [5, 10, 21, 42, 63]

# All feature column names (populated by compute_features)
FEATURE_COLS: list[str] = []


def compute_features(
    panel: pd.DataFrame,
    lookbacks: list[int] | None = None,
) -> pd.DataFrame:
    """Build a comprehensive feature matrix from daily OHLCV panel data.

    Uses TA-Lib for indicator computation.  All features are computed from
    data available at close of day t-1 (shifted by 1) to prevent lookahead.

    Feature groups (~50 features per asset per day):
      - Trailing returns at multiple lookbacks
      - Realized volatility at multiple lookbacks
      - Volume ratios at multiple lookbacks
      - Trend: ADX, MACD, Aroon, linear regression slope
      - Momentum: RSI, Stochastic, CCI, Williams %R, MFI, ROC, UltOsc
      - Volatility: ATR, Bollinger bandwidth, NATR
      - Volume: OBV trend, AD line trend
      - Price structure: Bollinger %B, channel position

    Parameters
    ----------
    panel : pd.DataFrame
        Must have columns: symbol, ts, open, high, low, close, volume
    lookbacks : list[int]
        Lookback windows for return/vol features. Default: [5, 10, 21, 42, 63]

    Returns
    -------
    pd.DataFrame with original columns plus feature columns.
    """
    import talib

    if lookbacks is None:
        lookbacks = RETURN_LOOKBACKS

    def _per_sym(g: pd.DataFrame) -> pd.DataFrame:
        g = g.copy()

        # Extract numpy arrays for TA-Lib (requires float64)
        o = g["open"].values.astype(np.float64)
        h = g["high"].values.astype(np.float64)
        lo = g["low"].values.astype(np.float64)
        c = g["close"].values.astype(np.float64)
        v = g["volume"].values.astype(np.float64)

        log_ret = np.log(g["close"] / g["close"].shift(1))

        # --- Returns and volatility at multiple lookbacks ---
        for lb in lookbacks:
            g[f"ret_{lb}d"] = g["close"].shift(1) / g["close"].shift(1 + lb) - 1.0
            g[f"vol_{lb}d"] = (
                log_ret.shift(1).rolling(lb, min_periods=lb).std()
                * np.sqrt(ANN_FACTOR)
            )
            vol_ma = g["volume"].shift(1).rolling(lb, min_periods=lb).mean()
            g[f"vol_ratio_{lb}d"] = g["volume"].shift(1) / vol_ma.clip(lower=1.0)

        # --- Trend indicators ---
        g["adx_14"] = talib.ADX(h, lo, c, timeperiod=14)
        g["adx_28"] = talib.ADX(h, lo, c, timeperiod=28)
        macd, macd_sig, macd_hist = talib.MACD(c, fastperiod=12, slowperiod=26, signalperiod=9)
        g["macd"] = macd
        g["macd_signal"] = macd_sig
        g["macd_hist"] = macd_hist
        aroon_down, aroon_up = talib.AROON(h, lo, timeperiod=14)
        g["aroon_osc"] = aroon_up - aroon_down
        g["linearreg_slope_14"] = talib.LINEARREG_SLOPE(c, timeperiod=14)
        g["linearreg_slope_42"] = talib.LINEARREG_SLOPE(c, timeperiod=42)

        # --- Momentum / oscillators ---
        g["rsi_14"] = talib.RSI(c, timeperiod=14)
        g["rsi_28"] = talib.RSI(c, timeperiod=28)
        slowk, slowd = talib.STOCH(h, lo, c, fastk_period=14, slowk_period=3, slowd_period=3)
        g["stoch_k"] = slowk
        g["stoch_d"] = slowd
        g["cci_14"] = talib.CCI(h, lo, c, timeperiod=14)
        g["cci_28"] = talib.CCI(h, lo, c, timeperiod=28)
        g["willr_14"] = talib.WILLR(h, lo, c, timeperiod=14)
        g["mfi_14"] = talib.MFI(h, lo, c, v, timeperiod=14)
        g["roc_10"] = talib.ROC(c, timeperiod=10)
        g["roc_21"] = talib.ROC(c, timeperiod=21)
        g["ultosc"] = talib.ULTOSC(h, lo, c, timeperiod1=7, timeperiod2=14, timeperiod3=28)

        # --- Volatility indicators ---
        g["atr_14"] = talib.ATR(h, lo, c, timeperiod=14)
        g["natr_14"] = talib.NATR(h, lo, c, timeperiod=14)
        upper, mid, lower = talib.BBANDS(c, timeperiod=20, nbdevup=2, nbdevdn=2)
        g["bb_width"] = (upper - lower) / np.where(mid > 0, mid, np.nan)
        g["bb_pctb"] = (c - lower) / np.where((upper - lower) > 0, upper - lower, np.nan)
        g["hl_range"] = (h - lo) / np.where(c > 0, c, np.nan)

        # --- Volume indicators ---
        obv = talib.OBV(c, v)
        g["obv_slope_14"] = talib.LINEARREG_SLOPE(obv, timeperiod=14)
        ad = talib.AD(h, lo, c, v)
        g["ad_slope_14"] = talib.LINEARREG_SLOPE(ad, timeperiod=14)

        # --- Price structure ---
        g["channel_pos_14"] = (c - talib.MIN(lo, 14)) / np.where(
            (talib.MAX(h, 14) - talib.MIN(lo, 14)) > 0,
            talib.MAX(h, 14) - talib.MIN(lo, 14),
            np.nan,
        )
        g["channel_pos_42"] = (c - talib.MIN(lo, 42)) / np.where(
            (talib.MAX(h, 42) - talib.MIN(lo, 42)) > 0,
            talib.MAX(h, 42) - talib.MIN(lo, 42),
            np.nan,
        )

        # --- Open-price features ---
        # Overnight gap: open vs prior close (captures sentiment shifts)
        prev_c = np.roll(c, 1)
        prev_c[0] = np.nan
        g["overnight_gap"] = (o - prev_c) / np.where(prev_c > 0, prev_c, np.nan)

        # Body ratio: (close - open) / (high - low) — bullish/bearish candle strength
        body = c - o
        full_range = h - lo
        g["body_ratio"] = body / np.where(full_range > 0, full_range, np.nan)

        # Upper/lower shadow ratios
        g["upper_shadow"] = (h - np.maximum(o, c)) / np.where(full_range > 0, full_range, np.nan)
        g["lower_shadow"] = (np.minimum(o, c) - lo) / np.where(full_range > 0, full_range, np.nan)

        # --- Candlestick patterns (TA-Lib, integer-coded: -100/0/+100) ---
        g["cdl_doji"] = talib.CDLDOJI(o, h, lo, c).astype(np.float64)
        g["cdl_hammer"] = talib.CDLHAMMER(o, h, lo, c).astype(np.float64)
        g["cdl_engulfing"] = talib.CDLENGULFING(o, h, lo, c).astype(np.float64)
        g["cdl_morningstar"] = talib.CDLMORNINGSTAR(o, h, lo, c).astype(np.float64)
        g["cdl_eveningstar"] = talib.CDLEVENINGSTAR(o, h, lo, c).astype(np.float64)
        g["cdl_3whitesoldiers"] = talib.CDL3WHITESOLDIERS(o, h, lo, c).astype(np.float64)
        g["cdl_3blackcrows"] = talib.CDL3BLACKCROWS(o, h, lo, c).astype(np.float64)

        # Dollar volume (useful for universe filtering / weighting)
        g["dollar_volume"] = g["close"] * g["volume"]

        # Shift all TA-Lib/derived features by 1 to prevent lookahead
        # (TA-Lib computes using data up to and including current bar)
        talib_cols = [
            "adx_14", "adx_28", "macd", "macd_signal", "macd_hist",
            "aroon_osc", "linearreg_slope_14", "linearreg_slope_42",
            "rsi_14", "rsi_28", "stoch_k", "stoch_d",
            "cci_14", "cci_28", "willr_14", "mfi_14",
            "roc_10", "roc_21", "ultosc",
            "atr_14", "natr_14", "bb_width", "bb_pctb", "hl_range",
            "obv_slope_14", "ad_slope_14",
            "channel_pos_14", "channel_pos_42",
            "overnight_gap", "body_ratio", "upper_shadow", "lower_shadow",
            "cdl_doji", "cdl_hammer", "cdl_engulfing",
            "cdl_morningstar", "cdl_eveningstar",
            "cdl_3whitesoldiers", "cdl_3blackcrows",
        ]
        for col in talib_cols:
            if col in g.columns:
                g[col] = g[col].shift(1)

        return g

    result = panel.groupby("symbol", group_keys=False).apply(_per_sym)

    # Populate FEATURE_COLS for downstream use
    base_feat = []
    for lb in lookbacks:
        base_feat.extend([f"ret_{lb}d", f"vol_{lb}d", f"vol_ratio_{lb}d"])
    base_feat.extend([
        "adx_14", "adx_28", "macd", "macd_signal", "macd_hist",
        "aroon_osc", "linearreg_slope_14", "linearreg_slope_42",
        "rsi_14", "rsi_28", "stoch_k", "stoch_d",
        "cci_14", "cci_28", "willr_14", "mfi_14",
        "roc_10", "roc_21", "ultosc",
        "atr_14", "natr_14", "bb_width", "bb_pctb", "hl_range",
        "obv_slope_14", "ad_slope_14",
        "channel_pos_14", "channel_pos_42",
        "overnight_gap", "body_ratio", "upper_shadow", "lower_shadow",
        "cdl_doji", "cdl_hammer", "cdl_engulfing",
        "cdl_morningstar", "cdl_eveningstar",
        "cdl_3whitesoldiers", "cdl_3blackcrows",
    ])
    FEATURE_COLS.clear()
    FEATURE_COLS.extend(base_feat)

    return result


def add_cross_sectional_ranks(
    panel: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    """Add cross-sectional percentile ranks for each feature at each date.

    For each feature column, adds a `{col}_xsrank` column with values in [0, 1].
    """
    df = panel.copy()
    for col in feature_cols:
        df[f"{col}_xsrank"] = df.groupby("ts")[col].rank(pct=True)
    return df


# ---------------------------------------------------------------------------
# Walk-forward split utilities
# ---------------------------------------------------------------------------
def walk_forward_splits(
    dates: pd.DatetimeIndex | np.ndarray,
    train_days: int = 365 * 2,
    test_days: int = 63,
    step_days: int = 63,
    min_train_days: int = 365,
) -> list[dict]:
    """Generate walk-forward train/test date splits.

    Parameters
    ----------
    dates : array of datetime
        Sorted unique dates in the dataset.
    train_days : int
        Number of days in each training window.
    test_days : int
        Number of days in each test window.
    step_days : int
        Step forward between successive splits.
    min_train_days : int
        Minimum training window size (for early splits).

    Returns
    -------
    list of dict with keys: train_start, train_end, test_start, test_end
    """
    dates = pd.DatetimeIndex(sorted(dates))
    splits = []
    i = 0
    while True:
        test_end_idx = min_train_days + i * step_days + test_days
        if test_end_idx > len(dates):
            break
        train_start = dates[max(0, min_train_days + i * step_days - train_days)]
        train_end = dates[min_train_days + i * step_days - 1]
        test_start = dates[min_train_days + i * step_days]
        test_end = dates[min(test_end_idx - 1, len(dates) - 1)]

        if test_start > train_end:
            splits.append({
                "train_start": train_start,
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end,
                "fold": i,
            })
        i += 1
    return splits


# ---------------------------------------------------------------------------
# ML evaluation utilities
# ---------------------------------------------------------------------------
def evaluate_predictions(
    y_true: pd.Series,
    y_pred: pd.Series,
) -> dict:
    """Compute standard ML evaluation metrics for return predictions.

    Returns
    -------
    dict with keys: ic (information coefficient / Spearman rank corr),
        pearson_corr, rmse, mae, hit_rate (directional accuracy),
        n_obs
    """
    from scipy import stats as sp_stats

    mask = y_true.notna() & y_pred.notna()
    yt = y_true[mask]
    yp = y_pred[mask]
    n = len(yt)

    if n < 10:
        return {k: np.nan for k in [
            "ic", "pearson_corr", "rmse", "mae", "hit_rate", "n_obs"
        ]}

    ic = float(sp_stats.spearmanr(yt, yp).statistic)
    pearson = float(np.corrcoef(yt, yp)[0, 1])
    rmse = float(np.sqrt(((yt - yp) ** 2).mean()))
    mae = float((yt - yp).abs().mean())
    hit_rate = float(((yt > 0) == (yp > 0)).mean())

    return {
        "ic": ic,
        "pearson_corr": pearson,
        "rmse": rmse,
        "mae": mae,
        "hit_rate": hit_rate,
        "n_obs": n,
    }
