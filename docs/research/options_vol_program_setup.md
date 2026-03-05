# Options & Volatility Program — Setup Guide

## Overview

This document covers the end-to-end setup for pulling options data from Interactive Brokers into our research and trading infrastructure. The options program adds three new modules to the codebase:

| Module | Location | Purpose |
|--------|----------|---------|
| **Volatility** | `src/volatility/` | Realized vol estimators, vol surface representation |
| **Pricing** | `src/pricing/` | Black-Scholes, Black-76, Bachelier models with Greeks |
| **Options Data** | `src/data/options/` | IB API chain fetching, vol surface snapshotting, DuckDB storage |

---

## 1. Prerequisites

### Install dependencies

```bash
cd /path/to/trend_crypto
pip install -e ".[options]"
```

This installs `ib_insync`, the async-friendly Python wrapper for IB's TWS API. The core pricing and volatility modules only require `scipy`, `numpy`, and `pandas` (already in the base dependencies).

### Interactive Brokers setup

You need one of the following running locally:

| Application | Paper Port | Live Port | Notes |
|-------------|-----------|-----------|-------|
| **TWS (Trader Workstation)** | 7497 | 7496 | Full GUI, heavier |
| **IB Gateway** | 4002 | 4001 | Headless, recommended for automation |

**TWS/Gateway configuration checklist:**

1. Open TWS or IB Gateway and log in.
2. Go to **File → Global Configuration → API → Settings**.
3. Check **Enable ActiveX and Socket Clients**.
4. Check **Allow connections from localhost only** (for security).
5. Set **Socket port** to 7497 (paper) or 7496 (live).
6. Uncheck **Read-Only API** (we need market data requests).
7. Under **Precautions**, uncheck **Bypass Order Precautions** unless you plan to route orders.

**Market data subscriptions required:**

For each asset class you want to trade, you need the corresponding market data subscription in your IB account:

| Asset Class | IB Subscription | Example Underlyings |
|-------------|----------------|---------------------|
| Commodities | NYMEX, COMEX, ICE | CL (crude), GC (gold), SI (silver), NG (natgas) |
| FX | Forex bundle | EUR, GBP, JPY, AUD (via FX options) |
| Interest Rates | CME, CBOT | ZN (10Y note), ZB (30Y bond), SR3 (SOFR) |
| Crypto | Crypto bundle | BTC, ETH (via Deribit or CME) |
| Equities/ETFs | US Securities Snapshot | SPY, QQQ, IWM, GLD, TLT |

Check your subscriptions at **Account → Settings → Market Data Subscriptions** in the IB portal.

---

## 2. Fetching Option Chains

Option chains define the available strikes and expiries for an underlying. Fetch them once and refresh periodically (e.g., weekly or when new expiries list).

### Futures options (commodities, rates)

```python
from data.options import IBOptionChainFetcher

fetcher = IBOptionChainFetcher(db_path="../data/market.duckdb", port=7497)
fetcher.connect()

# Crude oil — specify the futures contract month
fetcher.fetch_chain("CL", exchange="NYMEX", sec_type="FUT", last_trade_date="20260701")

# Gold
fetcher.fetch_chain("GC", exchange="COMEX", sec_type="FUT", last_trade_date="20260801")

# 10-Year Treasury Note
fetcher.fetch_chain("ZN", exchange="CBOT", sec_type="FUT", last_trade_date="20260901")

# SOFR 3-month
fetcher.fetch_chain("SR3", exchange="CME", sec_type="FUT", last_trade_date="20261201")

fetcher.disconnect()
```

### Equity / ETF options

```python
fetcher.connect()

fetcher.fetch_chain("SPY", exchange="SMART", sec_type="STK")
fetcher.fetch_chain("GLD", exchange="SMART", sec_type="STK")
fetcher.fetch_chain("TLT", exchange="SMART", sec_type="STK")

fetcher.disconnect()
```

### Inspecting stored chains

```python
# List all underlyings with stored chains
print(fetcher.list_underlyings())

# Summary for one underlying
print(fetcher.chain_summary("CL"))
# → {'underlying': 'CL', 'n_contracts': 4820, 'n_expiries': 12, ...}

# Load filtered chain as a Polars DataFrame
chain_df = fetcher.get_chain("CL", min_tte_days=7, max_tte_days=90)
print(chain_df.head())
```

---

## 3. Snapshotting Vol Surfaces

Vol surface snapshots capture the full implied vol surface at a point in time. These are the core data for vol research and trading.

### Taking a snapshot

```python
from data.options import IBVolSurfaceCollector

collector = IBVolSurfaceCollector(db_path="../data/market.duckdb", port=7497)
collector.connect()

# Snapshot crude oil vol surface
surface = collector.snapshot(
    "CL",
    exchange="NYMEX",
    sec_type="FUT",
    last_trade_date="20260701",
    max_expiries=6,           # nearest 6 expiries
    strike_range_pct=0.20,    # ±20% of spot
    min_tte_days=3,           # skip very near-term
    max_tte_days=180,         # up to 6 months
)

# The returned VolSurface object is ready for immediate analysis
print(surface.atm_term_structure())
print(surface.skew_term_structure())

collector.disconnect()
```

### What gets stored

Every snapshot writes to the `vol_surface_snaps` table in DuckDB:

| Column | Description |
|--------|-------------|
| `snap_ts` | Snapshot timestamp (UTC) |
| `underlying` | Symbol |
| `expiry` | Option expiry date |
| `tte_years` | Time to expiry in years |
| `strike` | Strike price |
| `moneyness` | ln(K/F) |
| `mid_iv` | Mid implied vol |
| `bid_iv` / `ask_iv` | Bid/ask implied vols |
| `delta`, `gamma`, `vega` | IB model Greeks |
| `forward` | Forward price |
| `volume`, `open_interest` | Liquidity metrics |

### Loading historical snapshots

```python
# List all snapshots for an underlying
snaps = collector.list_snapshots("CL")
print(snaps)

# Reconstruct a VolSurface from a stored snapshot
surface = collector.load_surface("CL")  # most recent
# or: surface = collector.load_surface("CL", snap_ts=specific_datetime)

# Use the surface
print(f"ATM 1M vol: {surface.nearest_slice(1/12).atm_iv():.1%}")
print(f"25d skew:   {surface.nearest_slice(1/12).skew_25d():.1%}")
```

---

## 4. Scheduling Automated Snapshots

For building a historical vol surface database, schedule periodic snapshots via cron or a script:

```python
"""scripts/snapshot_vol_surfaces.py"""
import sys
sys.path.insert(0, "src")

from data.options import IBVolSurfaceCollector

UNDERLYINGS = [
    ("CL", "NYMEX", "FUT", "20260701"),
    ("GC", "COMEX", "FUT", "20260801"),
    ("ZN", "CBOT", "FUT", "20260901"),
    ("SPY", "SMART", "STK", ""),
]

collector = IBVolSurfaceCollector(db_path="../data/market.duckdb", port=4002)
collector.connect()

for symbol, exchange, sec_type, ltd in UNDERLYINGS:
    try:
        surface = collector.snapshot(
            symbol, exchange=exchange, sec_type=sec_type,
            last_trade_date=ltd, max_expiries=8,
        )
        print(f"{symbol}: {len(surface.slices)} expiries snapped")
    except Exception as e:
        print(f"{symbol}: FAILED — {e}")

collector.close()
```

**Cron schedule** (US market hours, twice daily):

```bash
# 10:00 AM and 3:00 PM ET, Monday–Friday
0 10,15 * * 1-5 cd /path/to/trend_crypto && python scripts/snapshot_vol_surfaces.py
```

---

## 5. DuckDB Schema

The options data lives in the same `market.duckdb` database alongside spot OHLCV data. Three tables:

```sql
-- Static chain metadata (strikes, expiries, multipliers)
SELECT * FROM option_chains WHERE underlying = 'CL' LIMIT 5;

-- Point-in-time option quotes with Greeks
SELECT * FROM option_ticks WHERE underlying = 'CL' ORDER BY ts DESC LIMIT 10;

-- Aggregated vol surface snapshots
SELECT snap_ts, underlying, COUNT(*) AS n_ticks, COUNT(DISTINCT expiry) AS n_expiries
FROM vol_surface_snaps
GROUP BY 1, 2
ORDER BY snap_ts DESC;
```

Tables are created automatically on first use — no manual schema setup required.

---

## 6. Using the Pricing Engine

The pricing engine works independently of IB data and can be used immediately.

### Choosing the right model

| Model | Use For | Vol Convention |
|-------|---------|---------------|
| `BlackScholes` | Equity/ETF options, FX options, crypto spot options | Lognormal (%) |
| `Black76` | Futures options (commodities, rates, index futures) | Lognormal (%) |
| `Bachelier` | Interest rate options (swaptions, caps/floors, SOFR) | Normal (bps) |

### Quick pricing

```python
from pricing import BlackScholes, Black76, Bachelier

# SPY call: S=540, K=550, T=30d, vol=18%, r=5%
bs = BlackScholes()
g = bs.greeks(540, 550, 30/365, 0.18, 0.05, is_call=True)
print(f"Price: ${g.price:.2f}  Delta: {g.delta:.3f}  Theta: ${g.theta:.2f}/day")

# CL futures option: F=72, K=75, T=45d, vol=35%, r=5%
b76 = Black76()
g = b76.greeks(72, 75, 45/365, 0.35, 0.05, is_call=True)
print(f"Price: ${g.price:.2f}  Delta: {g.delta:.3f}  Vega: ${g.vega:.2f}/1%vol")

# SOFR swaption: F=4.50%, K=4.75%, T=90d, normal vol=80bps, r=5%
bach = Bachelier()
g = bach.greeks(4.50, 4.75, 90/365, 0.80, 0.05, is_call=False)
print(f"Price: {g.price:.4f}  Delta: {g.delta:.3f}")
```

### Implied vol from market prices

```python
from pricing import bs_iv, b76_iv

# What vol is the market implying?
iv = bs_iv(forward=540, strike=550, tte=30/365, price=8.50, rate=0.05, is_call=True)
print(f"Implied vol: {iv:.1%}")
```

---

## 7. Using the Volatility Estimators

These work on existing spot OHLCV data — no options data needed.

```python
from volatility import yang_zhang, compare_estimators, vol_cone

# Run all 5 estimators side-by-side
vols = compare_estimators(
    open_=df["open"], high=df["high"],
    low=df["low"], close=df["close"],
    window=20, ann_factor=365,
)
vols.plot(title="Realized Vol Estimators")

# Vol cone: is current vol rich or cheap?
cone = vol_cone(
    close=df["close"], high=df["high"],
    low=df["low"], open_=df["open"],
    ann_factor=365,
)
print(cone)
```

---

## 8. Research Notebook Sequence

| Notebook | Status | What It Does |
|----------|--------|-------------|
| `06_vol_estimators.ipynb` | Ready | Compare estimators, test forecasting power, vol cones |
| `07_vol_surface.ipynb` | Next | Build and analyze implied vol surfaces from IB snapshots |
| `08_vol_forecasting.ipynb` | Planned | Can we forecast realized vol better than implied? |
| `09_greeks_intuition.ipynb` | Planned | Interactive Greek visualization, P&L attribution |
| `10_convexity_backtest.ipynb` | Planned | Backtest vol strategies vs trend signals |
| `11_delta_hedging_sim.ipynb` | Planned | Simulate hedging costs, optimal rebalance frequency |

---

## 9. Troubleshooting

**"No module named ib_insync"**
```bash
pip install ib_insync
# or: pip install -e ".[options]"
```

**"Connection refused" on connect()**
- Verify TWS or IB Gateway is running.
- Check the port matches (7497 for TWS paper, 4002 for Gateway paper).
- In TWS: File → Global Configuration → API → Settings → Enable ActiveX and Socket Clients.

**"No option chains found"**
- For futures, you must specify `last_trade_date` (contract month).
- Check that your market data subscription covers this exchange.
- Some thinly-traded contracts may not have option chains.

**"Market data farm connection is broken"**
- IB's market data servers occasionally disconnect. The code will retry, but if persistent, restart TWS/Gateway.

**Snapshot returns empty VolSurface**
- `strike_range_pct` may be too narrow. Try 0.30 or 0.40.
- `min_tte_days` may be filtering out all expiries. Lower to 1.
- Check that you have a live market data subscription for options on this underlying.

**Rate limiting**
- IB limits to ~50 simultaneous market data requests.  The snapshot code rate-limits internally via `rate_limit_secs` (default 0.1s per contract).
- For large chains (100+ strikes), consider reducing `max_expiries` or narrowing `strike_range_pct`.
