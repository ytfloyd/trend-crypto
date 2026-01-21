#!/usr/bin/env python
"""
Alpha Engine - Convert heuristics into testable mathematical signals
Implements IC (Information Coefficient) testing framework for crypto alphas
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Optional, Dict, List
import warnings

warnings.filterwarnings('ignore')


class AlphaEngine:
    """
    Converts trading heuristics into testable alpha signals.

    Philosophy:
    - A heuristic is a "rule of thumb" (e.g., "Buy when momentum is strong")
    - An Alpha is a testable vector that represents that rule mathematically
    - IC (Information Coefficient) measures signal-to-future-return correlation

    Usage:
        engine = AlphaEngine(df)
        alphas = engine.get_alphas()
        ic_results = engine.compute_ic(alphas, forward_returns)
    """

    def __init__(self, df: pd.DataFrame):
        """
        Args:
            df: DataFrame with columns ['ts', 'symbol', 'open', 'high', 'low', 'close', 'volume']
                Index should be MultiIndex with (symbol, ts)
        """
        self.df = df.copy()

        # Ensure proper index
        if not isinstance(self.df.index, pd.MultiIndex):
            if 'symbol' in self.df.columns and 'ts' in self.df.columns:
                self.df = self.df.set_index(['symbol', 'ts'])
            else:
                raise ValueError("DataFrame must have MultiIndex (symbol, ts) or columns [symbol, ts]")

    def z_score(self, series: pd.Series, window: int = 60) -> pd.Series:
        """
        Standardizes a signal to allow comparison across assets.
        Z-score = (value - mean) / std

        A z-score of +2 means "2 standard deviations above mean"
        """
        roll_mean = series.groupby(level='symbol').rolling(window=window).mean().reset_index(level=0, drop=True)
        roll_std = series.groupby(level='symbol').rolling(window=window).std().reset_index(level=0, drop=True)
        return (series - roll_mean) / (roll_std + 1e-8)

    def cs_rank(self, series: pd.Series) -> pd.Series:
        """
        Cross-sectional rank (0 to 1 scale).
        At each timestamp, ranks all symbols relative to each other.
        """
        return series.groupby(level='ts').rank(pct=True)

    # ==========================================
    # TREND / MOMENTUM ALPHAS
    # ==========================================

    def alpha_momentum(self, window: int = 20) -> pd.Series:
        """
        Simple Momentum: Returns over N days

        Heuristic: "Buy assets that have been going up"
        Math: (Price_t - Price_{t-N}) / Price_{t-N}
        """
        close = self.df['close']
        ret = close.groupby(level='symbol').pct_change(window)
        return self.cs_rank(ret)

    def alpha_vol_adjusted_momentum(self, ret_window: int = 20, vol_window: int = 20) -> pd.Series:
        """
        Volatility-Adjusted Momentum (Sharpe-Style)

        Heuristic: "Strong trend with low volatility is better than strong trend with high volatility"
        Math: Returns / Volatility

        This is essentially a rolling Sharpe ratio.
        """
        close = self.df['close']

        # Returns over ret_window
        returns = close.groupby(level='symbol').pct_change(ret_window)

        # Volatility over vol_window
        daily_ret = close.groupby(level='symbol').pct_change()
        vol = daily_ret.groupby(level='symbol').rolling(vol_window).std().reset_index(level=0, drop=True)

        # Sharpe-style signal
        signal = returns / (vol + 1e-8)
        return self.cs_rank(signal)

    def alpha_ema_fan(self, fast: int = 10, slow: int = 50, vol_window: int = 20) -> pd.Series:
        """
        EMA Fan / Trend Strength

        Heuristic: "Buy when fast EMA is far above slow EMA, normalized by volatility"
        Math: (EMA_fast - EMA_slow) / σ

        This captures "steep trend" while accounting for market volatility.
        """
        close = self.df['close']

        # EMAs per symbol - reset index to remove extra level from ewm
        ema_fast = close.groupby(level='symbol').ewm(span=fast, adjust=False).mean().reset_index(level=0, drop=True)
        ema_slow = close.groupby(level='symbol').ewm(span=slow, adjust=False).mean().reset_index(level=0, drop=True)

        # Volatility normalization
        daily_ret = close.groupby(level='symbol').pct_change()
        vol = daily_ret.groupby(level='symbol').rolling(vol_window).std().reset_index(level=0, drop=True)

        # Signal
        signal = (ema_fast - ema_slow) / (ema_slow * (vol + 1e-8))
        return self.cs_rank(signal)

    def alpha_acceleration(self, window: int = 10) -> pd.Series:
        """
        Momentum Acceleration (2nd derivative)

        Heuristic: "Buy when momentum is accelerating, not just strong"
        Math: Δ(Returns) - change in returns over time

        Captures "momentum of momentum"
        """
        close = self.df['close']
        returns = close.groupby(level='symbol').pct_change(window)
        acceleration = returns.groupby(level='symbol').diff(window)
        return self.cs_rank(acceleration)

    # ==========================================
    # MEAN REVERSION ALPHAS
    # ==========================================

    def alpha_mean_reversion(self, window: int = 20) -> pd.Series:
        """
        Mean Reversion Signal

        Heuristic: "Buy when price is below its recent average"
        Math: -1 * (Price - MA) / MA

        Negative signal because high deviation = sell signal.
        """
        close = self.df['close']
        ma = close.groupby(level='symbol').rolling(window).mean().reset_index(level=0, drop=True)

        # Invert: when price is high vs MA, signal is negative
        signal = -1 * (close - ma) / (ma + 1e-8)
        return self.cs_rank(signal)

    def alpha_rsi_divergence(self, window: int = 14) -> pd.Series:
        """
        RSI-Based Mean Reversion

        Heuristic: "Buy oversold (RSI < 30), sell overbought (RSI > 70)"
        Math: 50 - RSI (inverted so extreme values have strong signals)
        """
        close = self.df['close']
        delta = close.groupby(level='symbol').diff()

        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.groupby(level='symbol').rolling(window).mean().reset_index(level=0, drop=True)
        avg_loss = loss.groupby(level='symbol').rolling(window).mean().reset_index(level=0, drop=True)

        rs = avg_gain / (avg_loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))

        # Signal: 50 - RSI so oversold gives positive signal
        signal = 50 - rsi
        return self.cs_rank(signal)

    # ==========================================
    # VOLATILITY ALPHAS
    # ==========================================

    def alpha_vol_compression(self, current_window: int = 10, avg_window: int = 60) -> pd.Series:
        """
        Volatility Compression / Squeeze

        Heuristic: "Buy when volatility is low - often precedes breakouts"
        Math: -1 * (Vol_current / Vol_avg)

        Low current vol gives positive signal.
        """
        close = self.df['close']
        daily_ret = close.groupby(level='symbol').pct_change()

        vol_current = daily_ret.groupby(level='symbol').rolling(current_window).std().reset_index(level=0, drop=True)
        vol_avg = daily_ret.groupby(level='symbol').rolling(avg_window).std().reset_index(level=0, drop=True)

        # Invert: low vol gives positive signal
        signal = -1 * (vol_current / (vol_avg + 1e-8))
        return self.cs_rank(signal)

    def alpha_vol_breakout(self, window: int = 20, threshold: float = 1.5) -> pd.Series:
        """
        Volatility Breakout

        Heuristic: "Buy when volatility spikes above threshold"
        Math: (Vol_current / Vol_avg) - threshold
        """
        close = self.df['close']
        daily_ret = close.groupby(level='symbol').pct_change()

        vol_current = daily_ret.groupby(level='symbol').rolling(5).std().reset_index(level=0, drop=True)
        vol_avg = daily_ret.groupby(level='symbol').rolling(window).std().reset_index(level=0, drop=True)

        signal = (vol_current / (vol_avg + 1e-8)) - threshold
        return self.cs_rank(signal)

    # ==========================================
    # VOLUME ALPHAS
    # ==========================================

    def alpha_volume_momentum(self, window: int = 20) -> pd.Series:
        """
        Volume Momentum

        Heuristic: "Buy when volume is increasing (accumulation)"
        Math: (Volume_current - Volume_MA) / Volume_MA
        """
        volume = self.df['volume']
        vol_ma = volume.groupby(level='symbol').rolling(window).mean().reset_index(level=0, drop=True)

        signal = (volume - vol_ma) / (vol_ma + 1e-8)
        return self.cs_rank(signal)

    def alpha_price_volume_trend(self, window: int = 10) -> pd.Series:
        """
        Price-Volume Correlation

        Heuristic: "Strong trends have volume in direction of price"
        Math: Correlation(Price Change, Volume) over rolling window
        """
        close = self.df['close']
        volume = self.df['volume']

        price_change = close.groupby(level='symbol').pct_change()

        def rolling_corr(group):
            return group['price'].rolling(window).corr(group['volume'])

        df_temp = pd.DataFrame({'price': price_change, 'volume': volume})
        signal = df_temp.groupby(level='symbol').apply(rolling_corr).reset_index(level=0, drop=True)

        return self.cs_rank(signal)

    # ==========================================
    # PAIR / RELATIVE STRENGTH ALPHAS
    # ==========================================

    def alpha_relative_strength(self, benchmark_symbol: str = 'BTC-USD', window: int = 20) -> pd.Series:
        """
        Relative Strength vs Benchmark

        Heuristic: "Buy altcoins that are outperforming BTC"
        Math: Returns_alt / Returns_BTC
        """
        close = self.df['close']

        # Get benchmark returns
        if benchmark_symbol in self.df.index.get_level_values('symbol'):
            btc_close = close.xs(benchmark_symbol, level='symbol')
            btc_returns = btc_close.pct_change(window)

            # Align with multi-index
            btc_returns_aligned = self.df.index.get_level_values('ts').map(btc_returns.to_dict())

            # Asset returns
            asset_returns = close.groupby(level='symbol').pct_change(window)

            # Relative strength
            signal = asset_returns / (btc_returns_aligned + 1e-8)
            return self.cs_rank(signal)
        else:
            return pd.Series(0, index=self.df.index)

    # ==========================================
    # MASTER FUNCTION
    # ==========================================

    def get_alphas(self, alpha_list: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Calculate all (or selected) alphas and return as DataFrame.

        Args:
            alpha_list: List of alpha names to compute. If None, computes all.

        Returns:
            DataFrame with MultiIndex (symbol, ts) and alpha columns
        """
        available_alphas = {
            'momentum': self.alpha_momentum,
            'vol_adj_momentum': self.alpha_vol_adjusted_momentum,
            'ema_fan': self.alpha_ema_fan,
            'acceleration': self.alpha_acceleration,
            'mean_reversion': self.alpha_mean_reversion,
            'rsi_divergence': self.alpha_rsi_divergence,
            'vol_compression': self.alpha_vol_compression,
            'vol_breakout': self.alpha_vol_breakout,
            'volume_momentum': self.alpha_volume_momentum,
            'price_volume_trend': self.alpha_price_volume_trend,
            'relative_strength': self.alpha_relative_strength,
        }

        if alpha_list is None:
            alpha_list = list(available_alphas.keys())

        results = {}
        for alpha_name in alpha_list:
            if alpha_name in available_alphas:
                print(f"Computing alpha: {alpha_name}...")
                try:
                    results[f'alpha_{alpha_name}'] = available_alphas[alpha_name]()
                except Exception as e:
                    print(f"  ERROR: {e}")
                    results[f'alpha_{alpha_name}'] = pd.Series(np.nan, index=self.df.index)

        # Combine into DataFrame
        alpha_df = pd.DataFrame(results)

        # Add forward returns for IC testing
        close = self.df['close']
        alpha_df['forward_return_1d'] = close.groupby(level='symbol').pct_change().shift(-1)
        alpha_df['forward_return_5d'] = close.groupby(level='symbol').pct_change(5).shift(-5)
        alpha_df['forward_return_10d'] = close.groupby(level='symbol').pct_change(10).shift(-10)

        return alpha_df.dropna()

    # ==========================================
    # IC ANALYSIS
    # ==========================================

    @staticmethod
    def compute_ic(alpha_df: pd.DataFrame, forward_return_col: str = 'forward_return_1d') -> pd.DataFrame:
        """
        Compute Information Coefficient for each alpha.

        IC = Correlation(Alpha_t, Return_{t+1})

        IC > 0.05: Excellent edge
        IC ~ 0.01-0.03: Tradable but weak
        IC < 0: Flip the signal

        Args:
            alpha_df: DataFrame from get_alphas()
            forward_return_col: Which forward return to test against

        Returns:
            DataFrame with IC statistics per alpha
        """
        alpha_cols = [col for col in alpha_df.columns if col.startswith('alpha_')]

        results = []
        for alpha_col in alpha_cols:
            # Overall IC
            ic = alpha_df[alpha_col].corr(alpha_df[forward_return_col])

            # IC by time period (to test stability)
            ic_by_date = alpha_df.groupby(level='ts').apply(
                lambda x: x[alpha_col].corr(x[forward_return_col])
            )
            ic_mean = float(ic_by_date.mean()) if not ic_by_date.empty else 0.0
            ic_std = float(ic_by_date.std()) if not ic_by_date.empty else 0.0

            # T-stat
            n = len(ic_by_date.dropna())
            if ic_std == 0 or n == 0 or np.isnan(ic_std):
                t_stat = 0.0
            else:
                t_stat = ic_mean / (ic_std / np.sqrt(n))

            results.append({
                'alpha': alpha_col,
                'ic': ic,
                'ic_mean': ic_mean,
                'ic_std': ic_std,
                't_stat': t_stat,
                'n_days': n
            })

        ic_df = pd.DataFrame(results).sort_values('ic', ascending=False)
        return ic_df

    @staticmethod
    def quantile_analysis(alpha_df: pd.DataFrame, alpha_col: str,
                         forward_return_col: str = 'forward_return_1d',
                         n_quantiles: int = 5) -> pd.DataFrame:
        """
        Quantile Spread Analysis.

        Split data into N buckets based on alpha value.
        Test: Does Q5 (top quintile) outperform Q1 (bottom quintile)?

        Args:
            alpha_df: DataFrame from get_alphas()
            alpha_col: Which alpha to test
            forward_return_col: Which forward return to test
            n_quantiles: Number of buckets (5 = quintiles)

        Returns:
            DataFrame with avg returns per quantile
        """
        # Assign quantile labels
        alpha_df['quantile'] = pd.qcut(alpha_df[alpha_col], q=n_quantiles, labels=False, duplicates='drop')

        # Average returns per quantile
        quantile_results = alpha_df.groupby('quantile')[forward_return_col].agg([
            'mean', 'std', 'count'
        ]).reset_index()

        quantile_results.columns = ['quantile', 'avg_return', 'std_return', 'n_obs']

        # Spread: Q5 - Q1
        if len(quantile_results) >= 2:
            top = quantile_results.iloc[-1]['avg_return']
            bottom = quantile_results.iloc[0]['avg_return']
            spread = top - bottom
            quantile_results['spread_vs_q1'] = quantile_results['avg_return'] - bottom
            print(f"\n{alpha_col} Spread (Q5 - Q1): {spread:.4%}")

        return quantile_results


# ==========================================
# HELPER: LOAD DATA FROM DUCKDB
# ==========================================

def load_crypto_data_from_duckdb(db_path: str, table: str = 'bars_1d_usd_universe_clean_adv10m',
                                 start: str = '2023-01-01', end: str = '2025-01-01',
                                 symbols: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Load crypto data from DuckDB for alpha testing.

    Args:
        db_path: Path to DuckDB file
        table: Table name (e.g., bars_1d_usd_universe_clean_adv10m)
        start: Start date
        end: End date
        symbols: Optional list of symbols to filter

    Returns:
        DataFrame with MultiIndex (symbol, ts)
    """
    import duckdb

    con = duckdb.connect(str(db_path))
    con.execute("SET TimeZone='UTC';")

    where = [f"ts >= '{start}'", f"ts <= '{end}'"]
    if symbols:
        symbol_list = "', '".join(symbols)
        where.append(f"symbol IN ('{symbol_list}')")

    where_clause = "WHERE " + " AND ".join(where)

    query = f"""
        SELECT ts, symbol, open, high, low, close, volume
        FROM {table}
        {where_clause}
        ORDER BY ts, symbol;
    """

    df = con.execute(query).fetch_df()
    con.close()

    df['ts'] = pd.to_datetime(df['ts'])
    df = df.set_index(['symbol', 'ts'])

    return df


if __name__ == "__main__":
    print("Alpha Engine - Convert Heuristics to Testable Signals")
    print("=" * 60)
    print("\nUsage:")
    print("  from alpha_engine_v0 import AlphaEngine, load_crypto_data_from_duckdb")
    print("  df = load_crypto_data_from_duckdb('../data/market.duckdb')")
    print("  engine = AlphaEngine(df)")
    print("  alphas = engine.get_alphas()")
    print("  ic_results = engine.compute_ic(alphas)")
    print("  print(ic_results)")
