"""
Entry point:  python -m scripts.research.sornette_lppl

Runs the full pipeline: bubble scan → signals → portfolio backtest.
"""
from .run_bubble_scan import main as scan_main
from .run_portfolio import main as portfolio_main

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("PHASE 1: BUBBLE SCAN")
    print("=" * 70)
    scan_main()

    print("\n" + "=" * 70)
    print("PHASE 2: PORTFOLIO BACKTEST")
    print("=" * 70)
    portfolio_main()
