#!/usr/bin/env python
"""
Example: Alpha Discovery Workflow
Shows end-to-end process from heuristic → IC testing → selection
"""
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from alpha_engine_v0 import AlphaEngine, load_crypto_data_from_duckdb

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 120)


def example_workflow():
    """
    Complete workflow: Load data → Test alphas → Select best signals
    """

    print("=" * 80)
    print("EXAMPLE: ALPHA DISCOVERY WORKFLOW")
    print("=" * 80)

    # ==========================================
    # STEP 1: Define Your Trading Heuristic
    # ==========================================
    print("\n[STEP 1] Trading Heuristic:")
    print("  'I notice that cryptocurrencies with strong recent momentum")
    print("   AND low volatility tend to continue rising.'")
    print("\n  Question: How do I test if this actually works?")
    print("  Answer: Convert it to an Alpha and measure IC!")

    # ==========================================
    # STEP 2: Load Historical Data
    # ==========================================
    print("\n[STEP 2] Loading historical data...")

    # For example purposes, use small dataset
    test_symbols = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'AVAX-USD', 'LINK-USD']

    df = load_crypto_data_from_duckdb(
        db_path='../data/market.duckdb',
        table='bars_1d_usd_universe_clean_adv10m',
        start='2024-01-01',
        end='2025-01-01',
        symbols=test_symbols
    )

    print(f"  ✓ Loaded {len(df)} daily bars")
    print(f"  ✓ Symbols: {', '.join(test_symbols)}")
    print(f"  ✓ Period: {df.index.get_level_values('ts').min().date()} to {df.index.get_level_values('ts').max().date()}")

    # ==========================================
    # STEP 3: Compute Alpha (Vol-Adjusted Momentum)
    # ==========================================
    print("\n[STEP 3] Computing alpha signal...")
    print("  Formula: alpha_vol_adj_momentum = (20d_returns) / (20d_volatility)")

    engine = AlphaEngine(df)

    # Compute just this one alpha
    alpha_df = engine.get_alphas(alpha_list=['vol_adj_momentum'])

    print(f"  ✓ Generated alpha for {len(alpha_df)} observations")

    # Show sample values
    print("\n  Sample Alpha Values (last 5 days, ETH):")
    sample = alpha_df.xs('ETH-USD', level='symbol').tail(5)[['alpha_vol_adj_momentum']]
    print(sample.to_string())

    # ==========================================
    # STEP 4: Compute Information Coefficient
    # ==========================================
    print("\n[STEP 4] Computing Information Coefficient (IC)...")
    print("  IC = Correlation(Alpha_today, Return_tomorrow)")

    ic_results = AlphaEngine.compute_ic(alpha_df, forward_return_col='forward_return_1d')

    print("\n  IC RESULTS:")
    print(f"    IC:        {ic_results.iloc[0]['ic']:.4f}")
    print(f"    IC Mean:   {ic_results.iloc[0]['ic_mean']:.4f}")
    print(f"    IC Std:    {ic_results.iloc[0]['ic_std']:.4f}")
    print(f"    T-Stat:    {ic_results.iloc[0]['t_stat']:.2f}")
    print(f"    N Days:    {ic_results.iloc[0]['n_days']:.0f}")

    # Interpretation
    ic = ic_results.iloc[0]['ic']
    if ic > 0.05:
        verdict = "✅ EXCELLENT - Strong predictive signal!"
    elif ic > 0.02:
        verdict = "✓ TRADABLE - Viable signal with good execution"
    elif ic > 0:
        verdict = "⚠️  WEAK - Barely profitable"
    else:
        verdict = "❌ WRONG DIRECTION - Consider flipping signal (multiply by -1)"

    print(f"\n  Verdict: {verdict}")

    # ==========================================
    # STEP 5: Quantile Spread Analysis
    # ==========================================
    print("\n[STEP 5] Quantile Spread Analysis...")
    print("  Question: Does Q5 (strong signal) actually outperform Q1 (weak signal)?")

    quantiles = AlphaEngine.quantile_analysis(
        alpha_df,
        alpha_col='alpha_vol_adj_momentum',
        forward_return_col='forward_return_1d',
        n_quantiles=5
    )

    print("\n  QUINTILE RETURNS:")
    print(quantiles[['quantile', 'avg_return', 'std_return', 'n_obs']].to_string(index=False))

    spread = quantiles.iloc[-1]['avg_return'] - quantiles.iloc[0]['avg_return']
    print(f"\n  Spread (Q5 - Q1): {spread:.2%} per day")

    if spread > 0:
        print(f"  ✓ Monotonic: Higher signal → Higher returns")
    else:
        print(f"  ✗ Non-monotonic: Signal may not be actionable")

    # ==========================================
    # STEP 6: Decision Framework
    # ==========================================
    print("\n[STEP 6] Trading Decision:")

    if ic > 0.03 and spread > 0:
        print("  🚀 TRADE THIS SIGNAL")
        print("     - High IC (>0.03)")
        print("     - Positive quantile spread")
        print("     - Add to your alpha ensemble!")
    elif ic > 0.01 and spread > 0:
        print("  ✓ POTENTIALLY TRADABLE")
        print("    - Test with transaction costs")
        print("    - May work as part of ensemble")
    else:
        print("  ⚠️  DO NOT TRADE")
        print("    - IC too low or negative spread")
        print("    - Try flipping signal or different parameters")

    # ==========================================
    # SUMMARY
    # ==========================================
    print("\n" + "=" * 80)
    print("WORKFLOW COMPLETE")
    print("=" * 80)
    print("\nWhat you learned:")
    print("  1. Converted heuristic → mathematical signal (alpha)")
    print("  2. Measured predictive power (IC)")
    print("  3. Validated with quantile spreads")
    print("  4. Made data-driven decision (trade or not)")
    print("\nThis is how quantitative desks operate - no vibes, just math!")
    print("=" * 80)


if __name__ == "__main__":
    example_workflow()
