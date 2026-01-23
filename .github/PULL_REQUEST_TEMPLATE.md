# Engine: Model B open-to-close timing + funding diagnostics + strict_validation-gated asserts

## What

- Standardizes backtest timing to Model B (open-to-close) and adds funding diagnostics.

## Why

- Momentum/trend strategies are sensitive to breakout candle moves; open-to-close captures the executed-bar move correctly.
- Funding is a first-order PnL component for perpetual futures; must be visible in outputs and summaries.
- Defensive checks should not slow research sweeps.

## Changes

### Engine (`src/backtest/engine.py`)

- **Timing Model B (open-to-close):**
  - Added `compute_asset_returns()` helper that computes `asset_ret[t] = close[t] / open[t] - 1` when `open` column exists.
  - Falls back to close-to-close (`close[t] / close[t-1] - 1`) when `open` is missing.
  - Records fallback usage in `summary["used_close_to_close_fallback"]`.
  - Position return calculation: `gross_ret = w_held * asset_ret` (no double shift).

- **Funding diagnostics:**
  - Added `funding_costs` and `cum_funding_costs` columns to `equity_df`.
  - Added summary fields:
    - `total_funding_cost` — total funding over backtest
    - `avg_funding_cost_per_bar` — average per-bar funding
    - `funding_cost_as_pct_of_gross` — funding as % of gross PnL magnitude (None if gross ≈ 0)
    - `funding_convention` — `"positive_means_longs_pay"` (metadata for data integration)
    - `return_mode` — `"open_to_close"` or `"close_to_close_fallback"` (explicit mode indicator)

- **Gated defensive assertions:**
  - Output-length assertions (funding_costs, nav) now only run when `cfg.engine.strict_validation=True`.
  - Avoids overhead in research sweeps.

- **Code quality:**
  - Replaced magic epsilon (`1e-9`) with named constant `_GROSS_PNL_EPSILON`.

### Docs (`README.md`)

- **Added "Timing & Returns" section** documenting:
  - Signal decision at `Close(t)`, execution at `Open(t + execution_lag_bars)`.
  - PnL uses open→close by default (Model B).
  - Close→close fallback behavior and `used_close_to_close_fallback` flag.
  - "No double shift" property: positions shifted exactly once for execution lag.

- **Added "Funding" section** documenting:
  - Funding cost formula: `funding_costs[t] = position[t] * funding_rate[t]`.
  - Sign convention: `positive_means_longs_pay` (Binance/Bybit style).
  - All funding outputs: per-bar, cumulative, summary fields.
  - Sign verification guidance via cumulative funding curves.

### Tests

- **New test file:** `tests/test_open_to_close_timing.py`
  - `test_compute_asset_returns_open_to_close()` — verifies open-to-close calculation
  - `test_compute_asset_returns_fallback_to_close_to_close()` — verifies fallback when `open` missing
  - `test_open_to_close_timing_model_b()` — **critical test that fails with old double-shift bug**
  - `test_funding_diagnostics_present()` — verifies funding columns and summary fields

- **Updated test helper:** `tests/test_engine_costs.py`
  - Fixed `_bars()` to create realistic OHLC where `open[t] = close[t-1]` for open-to-close compatibility.

## Test Plan

```bash
# Run full test suite
pytest -q

# Run timing-specific tests
pytest tests/test_open_to_close_timing.py -v
pytest tests/test_engine_costs.py -v
pytest tests/test_temporal_integrity.py -v
```

**Expected:** All 54 tests pass.

## Backward Compatibility

- **No breaking changes** to public APIs or function signatures.
- Existing backtests will automatically use open-to-close if `open` column exists; otherwise, close-to-close fallback is transparent.
- New summary fields are additive; existing summary consumers are unaffected.

## Deployment Readiness

- [x] Tests pass (`pytest -q`)
- [x] Documentation updated (README.md)
- [x] No external dependencies added
- [x] Behavior is deterministic and reproducible
- [x] Trade log disclaimer added (`binary_entries_exits_only`)

## Reviewer Checklist

- [ ] Timing model is correct (no lookahead, execution at open[t+lag])
- [ ] Funding sign convention matches venue docs (positive = longs pay)
- [ ] Defensive assertions only run in strict mode
- [ ] Documentation is clear and actionable
- [ ] Tests cover the critical path (open-to-close, fallback, funding presence)

## Follow-up (Out of Scope)

- Integrate actual funding rate data from Binance/Bybit APIs (currently placeholder 0.0).
- Add per-symbol funding diagnostics for multi-asset portfolios.
- Extend trade log to handle partial resizes (vol-targeting, cluster caps).
