# Medallion Lite — Execution Integration Guide

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     DATA LAYER                              │
│                                                             │
│  [Coinbase API] ──→ [Collector cron] ──→ [DuckDB bars_1h]  │
│                      (every 5 min)                          │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                   SIGNAL SERVICE                            │
│                                                             │
│  MedallionSignalService.run_cycle()                         │
│    1. Load recent hourly bars (5000h lookback)              │
│    2. Compute ensemble regime score                         │
│    3. Compute cross-sectional factors + composite           │
│    4. Update holdings (exits: regime, stop, factor, maxhold)│
│    5. Evaluate entries (at rebalance points)                │
│    6. Compute target weights                                │
│    7. Publish                                               │
│                                                             │
│  State: medallion_state.json (persists between cycles)      │
│  Output: signal_output.json  (consumed by execution)        │
│  History: live_signals table in DuckDB                      │
└──────────────────────────┬──────────────────────────────────┘
                           │
              ┌────────────▼────────────┐
              │   SIGNAL CONTRACT       │
              │   signal_output.json    │
              │   (see schema below)    │
              └────────────┬────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                  EXECUTION ENGINE                           │
│                  (your team's system)                       │
│                                                             │
│  1. Read signal_output.json                                 │
│  2. Extract target_weights: {symbol: weight}                │
│  3. Compare to current positions                            │
│  4. Generate orders for deltas above deadband               │
│  5. Execute via TWAP/VWAP on Coinbase                       │
│  6. Reconcile positions post-fill                           │
└─────────────────────────────────────────────────────────────┘
```

## Signal Output Contract

The signal service writes `signal_output.json` every cycle. This is
the **only file the execution engine needs to read**.

```json
{
  "ts": "2026-03-17T14:00:00+00:00",
  "cycle_id": "a1b2c3d4",
  "target_weights": {
    "BTC-USD": 0.082,
    "ETH-USD": 0.095,
    "SOL-USD": 0.078,
    "AVAX-USD": 0.065,
    "DOGE-USD": 0.0,
    "...": 0.0
  },
  "regime_score": 0.72,
  "actions": [
    {
      "symbol": "AVAX-USD",
      "action": "entry",
      "target_weight": 0.065,
      "score": 0.78,
      "reason": "",
      "hours_held": 0,
      "cum_ret": 0.0
    },
    {
      "symbol": "DOGE-USD",
      "action": "exit_stop",
      "target_weight": 0.0,
      "reason": "",
      "hours_held": 72,
      "cum_ret": -0.12
    }
  ],
  "diagnostics": {
    "n_holdings": 7,
    "gross_exposure": 0.58,
    "n_eligible": 12,
    "data_freshness_hours": 0.5,
    "cycle_count": 1247,
    "is_rebalance": true
  },
  "stale": false
}
```

### Field Reference

| Field | Type | Description |
|-------|------|-------------|
| `ts` | ISO 8601 | UTC timestamp of signal computation |
| `cycle_id` | string | Unique ID for idempotency (don't execute same cycle twice) |
| `target_weights` | {symbol: float} | **THE KEY OUTPUT.** Target portfolio weight per symbol. Sum ≤ 1. Zero means "no position / close existing." |
| `regime_score` | float [0,1] | Current market regime probability. Below 0.15 = emergency flat. |
| `actions` | list | What changed this cycle (entries, exits). Informational for logging/audit. |
| `diagnostics` | dict | System health metrics |
| `stale` | bool | True if data is >3h old. **Execution should NOT trade on stale signals.** |

### Execution Engine Pseudocode

```python
import json

signal = json.load(open("signal_output.json"))

# Safety checks
if signal["stale"]:
    log.warning("Stale signal — skipping execution")
    return

if signal["cycle_id"] == last_executed_cycle_id:
    log.info("Already executed this cycle — skipping")
    return

# Extract target weights
target_weights = signal["target_weights"]
current_weights = get_current_portfolio_weights()

# Generate orders
for symbol in set(target_weights) | set(current_weights):
    target = target_weights.get(symbol, 0.0)
    current = current_weights.get(symbol, 0.0)
    delta = target - current

    if abs(delta) < DEADBAND:  # e.g. 1%
        continue

    if abs(delta) * NAV < MIN_TRADE_NOTIONAL:  # e.g. $50
        continue

    side = "BUY" if delta > 0 else "SELL"
    notional = abs(delta) * NAV
    submit_twap_order(symbol, side, notional)

last_executed_cycle_id = signal["cycle_id"]
```

## Deployment Options

### Option A: Cron (Recommended for v1)

```bash
# Signal service: runs at minute 5 of every hour
# (5 min after the hour to let the collector finish syncing)
5 * * * * cd /path/to/trend_crypto && python scripts/run_medallion_live.py \
    --state-dir /path/to/live_state >> /var/log/medallion_signal.log 2>&1

# Execution engine: runs at minute 10 of every hour
# (5 min after signal service to ensure fresh signals)
10 * * * * /path/to/execution_engine --signal-file /path/to/live_state/signal_output.json
```

### Option B: Daemon Mode

```bash
# Signal service runs continuously, cycling every hour
python scripts/run_medallion_live.py --daemon --interval 3600
```

### Option C: LiveRunner Integration

If your execution engine uses the existing `LiveRunner` + `OMS`:

```python
from live.medallion_signal import MedallionSignalService, SignalConfig
from strategy.medallion_portfolio import MedallionEmbeddedAdapter
from data.live_feed import DuckDBLiveDataFeed
from execution.oms import OrderManagementSystem

# Setup
config = SignalConfig(db_path="/path/to/market.duckdb")
signal_svc = MedallionSignalService(config)
adapter = MedallionEmbeddedAdapter(signal_svc)

# Or read from file (when signal service is a separate process):
from strategy.medallion_portfolio import MedallionPortfolioAdapter
adapter = MedallionPortfolioAdapter(signal_file="/path/to/signal_output.json")

# Use with OMS
weights = adapter.on_bar_close_portfolio(contexts)
result = oms.rebalance_to_targets(weights, current_weights, nav)
```

## Data Dependencies

| Dependency | Update Frequency | Source |
|-----------|-----------------|--------|
| Hourly OHLCV (bars_1h) | Continuous (collector) | Coinbase Advanced Trade API → DuckDB |
| BTC daily close | Derived from hourly | Resampled in signal service |
| Universe (50 tokens) | Computed each cycle | SQL median ADV filter |
| Portfolio state | Persisted each cycle | medallion_state.json |

### Required Lookback

The signal service needs **5,000 hours** (~208 days) of history to initialise:
- 200-day SMA for BTC regime: 4,800 hours
- 168-hour rolling windows for factors

After initial warmup, only the most recent bar is consumed each cycle.

## State Management

`medallion_state.json` persists between cycles:

```json
{
  "holdings": {
    "SOL-USD": {
      "symbol": "SOL-USD",
      "entry_ts": "2026-03-15T14:00:00+00:00",
      "entry_score": 0.78,
      "hours_held": 48,
      "cum_ret": 0.12,
      "peak_cum": 0.15
    }
  },
  "last_cycle_ts": "2026-03-17T14:00:00+00:00",
  "last_rebalance_hour": 1224,
  "cycle_count": 1248
}
```

**Important:** If the state file is deleted, the service starts fresh with
no holdings. This is safe but will miss any existing positions — coordinate
with the execution engine to reconcile.

## Monitoring & Alerts

| Condition | Action |
|-----------|--------|
| `stale: true` | Data pipeline is behind. Don't trade. Alert ops. |
| `regime_score < 0.15` | Emergency flat. All weights → 0. |
| `n_holdings == 0` for >48h | Possible issue with entry logic or stale data. |
| Signal file not updated for >2h | Signal service may have crashed. |
| `gross_exposure > 1.0` | Bug in sizing logic. Don't trade. |
| Execution drift > 5% from target | Reconciliation issue. |

## File Index

| File | Purpose |
|------|---------|
| `src/live/medallion_signal.py` | Signal service (core logic + state) |
| `src/strategy/medallion_portfolio.py` | PortfolioStrategy adapter for LiveRunner |
| `src/data/live_feed.py` | DuckDB-backed DataFeed for LiveRunner |
| `scripts/run_medallion_live.py` | CLI entry point (cron/daemon/paper) |
| `scripts/research/medallion_lite/` | Research backtest code (reference) |
