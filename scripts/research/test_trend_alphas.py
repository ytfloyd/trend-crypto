#!/usr/bin/env python
"""
Test Trend/Momentum Alphas - IC Analysis
Systematic alpha discovery using Information Coefficient testing
"""
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from alpha_engine_v0 import AlphaEngine, load_crypto_data_from_duckdb

sns.set_style('darkgrid')


def run_alpha_tests(db_path: str, output_dir: str = 'artifacts/research/alpha_tests'):
    """
    Run comprehensive alpha IC tests on crypto data.

    Args:
        db_path: Path to DuckDB file
        output_dir: Where to save results
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("ALPHA IC TESTING - TREND/MOMENTUM STRATEGY DISCOVERY")
    print("=" * 80)

    # ==========================================
    # 1. LOAD DATA
    # ==========================================
    print("\n[1/5] Loading crypto data from DuckDB...")

    # Use top liquid coins for testing
    test_symbols = [
        'BTC-USD', 'ETH-USD', 'SOL-USD', 'AVAX-USD', 'LINK-USD',
        'DOGE-USD', 'LTC-USD', 'XRP-USD', 'ADA-USD', 'MATIC-USD'
    ]

    df = load_crypto_data_from_duckdb(
        db_path=db_path,
        table='bars_1d_usd_universe_clean_adv10m',
        start='2023-01-01',
        end='2025-01-01',
        symbols=test_symbols
    )

    print(f"  Loaded {len(df)} rows across {len(test_symbols)} symbols")
    print(f"  Date range: {df.index.get_level_values('ts').min()} to {df.index.get_level_values('ts').max()}")

    # ==========================================
    # 2. COMPUTE ALPHAS
    # ==========================================
    print("\n[2/5] Computing alphas...")

    engine = AlphaEngine(df)
    alpha_df = engine.get_alphas()

    print(f"  Generated {len([c for c in alpha_df.columns if c.startswith('alpha_')])} alphas")
    print(f"  Total observations: {len(alpha_df)}")

    # Save raw alphas
    alpha_csv = output_path / 'alphas_raw.csv'
    alpha_df.to_csv(alpha_csv)
    print(f"  Saved: {alpha_csv}")

    # ==========================================
    # 3. IC ANALYSIS (1-day forward returns)
    # ==========================================
    print("\n[3/5] Computing Information Coefficients (1-day horizon)...")

    ic_1d = AlphaEngine.compute_ic(alpha_df, forward_return_col='forward_return_1d')
    print("\nIC Results (1-day forward return):")
    print(ic_1d.to_string(index=False))

    # Save
    ic_1d_csv = output_path / 'ic_results_1d.csv'
    ic_1d.to_csv(ic_1d_csv, index=False)
    print(f"\nSaved: {ic_1d_csv}")

    # ==========================================
    # 4. IC ANALYSIS (5-day and 10-day)
    # ==========================================
    print("\n[4/5] Computing IC for multi-day horizons...")

    ic_5d = AlphaEngine.compute_ic(alpha_df, forward_return_col='forward_return_5d')
    ic_10d = AlphaEngine.compute_ic(alpha_df, forward_return_col='forward_return_10d')

    print("\nIC Results (5-day forward return):")
    print(ic_5d[['alpha', 'ic', 't_stat']].head(10).to_string(index=False))

    print("\nIC Results (10-day forward return):")
    print(ic_10d[['alpha', 'ic', 't_stat']].head(10).to_string(index=False))

    # Save
    ic_5d.to_csv(output_path / 'ic_results_5d.csv', index=False)
    ic_10d.to_csv(output_path / 'ic_results_10d.csv', index=False)

    # ==========================================
    # 5. QUANTILE ANALYSIS (Top 3 Alphas)
    # ==========================================
    print("\n[5/5] Running Quantile Spread Analysis on top 3 alphas...")

    top_alphas = ic_1d.head(3)['alpha'].tolist()

    for alpha_col in top_alphas:
        print(f"\n--- {alpha_col} ---")
        quantiles = AlphaEngine.quantile_analysis(
            alpha_df,
            alpha_col=alpha_col,
            forward_return_col='forward_return_1d',
            n_quantiles=5
        )
        print(quantiles.to_string(index=False))

        # Save
        quantiles.to_csv(output_path / f'quantiles_{alpha_col}.csv', index=False)

    # ==========================================
    # 6. VISUALIZATION
    # ==========================================
    print("\n[6/6] Generating visualizations...")

    # IC Bar Chart
    fig, ax = plt.subplots(figsize=(12, 6))
    ic_1d_sorted = ic_1d.sort_values('ic', ascending=False)
    colors = ['green' if x > 0 else 'red' for x in ic_1d_sorted['ic']]

    ax.barh(ic_1d_sorted['alpha'], ic_1d_sorted['ic'], color=colors, alpha=0.7)
    ax.axvline(x=0, color='black', linestyle='--', linewidth=0.5)
    ax.axvline(x=0.05, color='green', linestyle=':', linewidth=0.5, label='IC > 0.05 (Excellent)')
    ax.axvline(x=-0.05, color='red', linestyle=':', linewidth=0.5, label='IC < -0.05 (Flip Signal)')
    ax.set_xlabel('Information Coefficient (1-day forward return)')
    ax.set_title('Alpha IC Rankings - Crypto Trend/Momentum Signals')
    ax.legend()
    plt.tight_layout()

    ic_chart = output_path / 'ic_chart_1d.png'
    plt.savefig(ic_chart, dpi=150)
    print(f"  Saved: {ic_chart}")
    plt.close()

    # IC Scatter: IC vs T-stat
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(ic_1d['ic'], ic_1d['t_stat'], s=100, alpha=0.6, c=ic_1d['ic'], cmap='RdYlGn')
    ax.axhline(y=3, color='green', linestyle='--', linewidth=0.5, label='t-stat > 3 (Significant)')
    ax.axhline(y=-3, color='red', linestyle='--', linewidth=0.5)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Information Coefficient')
    ax.set_ylabel('T-Statistic')
    ax.set_title('Alpha Quality Map: IC vs T-Stat')
    ax.legend()

    # Annotate top alphas
    for idx, row in ic_1d.head(5).iterrows():
        ax.annotate(row['alpha'].replace('alpha_', ''),
                   xy=(row['ic'], row['t_stat']),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=8, alpha=0.7)

    plt.colorbar(scatter, label='IC')
    plt.tight_layout()

    scatter_chart = output_path / 'ic_vs_tstat.png'
    plt.savefig(scatter_chart, dpi=150)
    print(f"  Saved: {scatter_chart}")
    plt.close()

    # ==========================================
    # SUMMARY REPORT
    # ==========================================
    print("\n" + "=" * 80)
    print("SUMMARY REPORT")
    print("=" * 80)

    print("\n📊 TOP 5 ALPHAS (by IC):")
    for idx, row in ic_1d.head(5).iterrows():
        status = "✅ EXCELLENT" if abs(row['ic']) > 0.05 else "⚠️  WEAK" if abs(row['ic']) < 0.02 else "✓ TRADABLE"
        sign = "LONG" if row['ic'] > 0 else "SHORT (FLIP)"
        print(f"  {row['alpha']:30s} | IC: {row['ic']:+.4f} | t-stat: {row['t_stat']:+.2f} | {status} | {sign}")

    print("\n📉 WORST 3 ALPHAS (consider flipping):")
    for idx, row in ic_1d.tail(3).iterrows():
        print(f"  {row['alpha']:30s} | IC: {row['ic']:+.4f} | t-stat: {row['t_stat']:+.2f}")

    print("\n📁 All results saved to:")
    print(f"  {output_path.resolve()}")

    print("\n" + "=" * 80)
    print("✅ ALPHA TESTING COMPLETE")
    print("=" * 80)

    return ic_1d, alpha_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Test trend/momentum alphas with IC analysis')
    parser.add_argument('--db', type=str, default='../data/market.duckdb',
                       help='Path to DuckDB file')
    parser.add_argument('--out', type=str, default='artifacts/research/alpha_tests',
                       help='Output directory')

    args = parser.parse_args()

    ic_results, alpha_df = run_alpha_tests(args.db, args.out)
