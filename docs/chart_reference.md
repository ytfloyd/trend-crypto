# Chart Tool — Quick Reference

```
python scripts/chart.py SYMBOL [options]
python scripts/chart.py --list          # show all overlays
```

Charts are **interactive** (Plotly-based) and open in your browser. Zoom, pan, hover for values, and use the range selector buttons to navigate.

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `SYMBOL` | *(required)* | Trading pair: `BTC-USD`, `SOL-USD`, `ETH-USD`, etc. |
| `--tf` | `1d` | Bar timeframe: `5m`, `30m`, `1h`, `4h`, `8h`, `1d` |
| `--days` | `180` | Lookback in calendar days |
| `--start` | | Explicit start date (ISO), overrides `--days` |
| `--end` | | Explicit end date (ISO), default today |
| `--overlay` | | Overlay spec (repeatable, see below) |
| `--candles` | off | Render OHLC candlesticks instead of close line |
| `--save` | | Save interactive `.html` file instead of opening browser |
| `--title` | auto | Custom chart title |
| `--db` | auto | Override DuckDB path |
| `--list` | | Print all available overlays and exit |

## Built-in Overlays

These work out of the box with no extra dependencies.

| Overlay | Syntax | Axis | Description |
|---------|--------|------|-------------|
| **Realized Vol** | `rvol:Nd` | Secondary (right) | Rolling close-to-close annualized vol, N in calendar days |
| **SMA** | `ma:N1,N2,...` | Price | Simple moving averages, N in bars |
| **EMA** | `ema:N1,N2,...` | Price | Exponential moving averages, N in bars |
| **Bollinger Bands** | `bb:N,K` | Price (shaded) | N-bar MA +/- K standard deviations |
| **Volume** | `vol` | Subplot (below) | Volume bars colored green/red by bar direction |
| **ATR** | `atr:N` | Secondary (right) | Average True Range over N bars |

## TA-Lib Overlays

Requires TA-Lib C library and Python wrapper:

```bash
brew install ta-lib        # macOS
pip install TA-Lib>=0.4.28
```

### Trend

| Overlay | Syntax | Axis | Default |
|---------|--------|------|---------|
| ADX | `adx:N` | Secondary | 14 |
| DX | `dx:N` | Secondary | 14 |
| +DI | `plus_di:N` | Secondary | 14 |
| -DI | `minus_di:N` | Secondary | 14 |
| MACD | `macd:fast,slow,sig` | Secondary (3 lines) | 12,26,9 |
| Aroon Osc | `aroon:N` | Secondary | 14 |
| LinReg Slope | `linearreg_slope:N` | Secondary | 14 |
| Parabolic SAR | `sar:accel,max` | Price | 0.02,0.2 |
| HT Trendline | `ht_trendline` | Price | — |
| KAMA | `kama:N` | Price | 10 |

### Momentum / Oscillators

| Overlay | Syntax | Axis | Default |
|---------|--------|------|---------|
| RSI | `rsi:N` | Secondary (0-100) | 14 |
| Slow Stochastic | `stoch:fastk,slowk,slowd` | Secondary (2 lines) | 14,3,3 |
| Fast Stochastic | `stochf:fastk,fastd` | Secondary (2 lines) | 14,3 |
| Stochastic RSI | `stochrsi:N,fastk,fastd` | Secondary (2 lines) | 14,5,3 |
| CCI | `cci:N` | Secondary | 14 |
| Williams %R | `willr:N` | Secondary | 14 |
| MFI | `mfi:N` | Secondary (0-100) | 14 |
| ROC | `roc:N` | Secondary | 10 |
| Momentum | `mom:N` | Secondary | 10 |
| Ultimate Osc | `ultosc:p1,p2,p3` | Secondary | 7,14,28 |
| APO | `apo:fast,slow` | Secondary | 12,26 |
| PPO | `ppo:fast,slow` | Secondary | 12,26 |
| CMO | `cmo:N` | Secondary | 14 |
| BOP | `bop` | Secondary | — |
| TRIX | `trix:N` | Secondary | 14 |

### Volatility

| Overlay | Syntax | Axis | Default |
|---------|--------|------|---------|
| Normalized ATR | `natr:N` | Secondary | 14 |
| BB Width | `bb_width:N` | Secondary | 20 |
| BB %B | `bb_pctb:N` | Secondary (0-1) | 20 |
| True Range | `trange` | Secondary | — |

### Volume

| Overlay | Syntax | Axis | Default |
|---------|--------|------|---------|
| OBV Slope | `obv:N` | Secondary | 14 |
| A/D Slope | `ad:N` | Secondary | 14 |

### Price Structure

| Overlay | Syntax | Axis | Default |
|---------|--------|------|---------|
| Donchian Pos | `donchian:N` | Secondary (0-1) | 14 |

## Examples

```bash
# Daily BTC with RSI
python scripts/chart.py BTC-USD --overlay rsi:14

# 4h SOL with MACD (3 lines: MACD, signal, histogram)
python scripts/chart.py SOL-USD --tf 4h --overlay macd:12,26,9

# 1h ETH with Slow Stochastic + volume
python scripts/chart.py ETH-USD --tf 1h --days 30 --overlay stoch --overlay vol

# Daily BTC with ADX + Parabolic SAR on price
python scripts/chart.py BTC-USD --overlay adx:14 --overlay sar

# Candlesticks with KAMA and Bollinger %B
python scripts/chart.py SOL-USD --tf 4h --candles --overlay kama:21 --overlay bb_pctb:20

# Kitchen sink: candlesticks, EMAs, RSI, volume, realized vol
python scripts/chart.py BTC-USD --tf 4h --days 90 --candles \
  --overlay ema:12,26 --overlay rsi:14 --overlay vol --overlay rvol:14d

# Donchian channel position with MFI
python scripts/chart.py ETH-USD --overlay donchian:20 --overlay mfi:14

# Normalized ATR vs realized vol
python scripts/chart.py SOL-USD --overlay natr:14 --overlay rvol:14d
```

## From a Notebook

```python
from chart import chart

chart("SOL-USD", tf="4h", overlays=["rsi:14"])
chart("BTC-USD", tf="1h", days=60, overlays=["macd", "vol"], candles=True)
chart("ETH-USD", overlays=["stoch", "adx:21", "sar"])
```

## Interactive Features

Charts render in your default browser as interactive Plotly figures.

| Action | How |
|--------|-----|
| **Zoom** | Click-drag to select a region, or use scroll wheel |
| **Pan** | Shift + drag |
| **Reset zoom** | Double-click the chart |
| **Hover** | Unified crosshair tooltip shows price + all overlay values |
| **Range buttons** | 1M, 3M, 6M, YTD, 1Y, All — above the chart |
| **Range slider** | Draggable minimap below the chart for quick navigation |
| **Save** | `--save path.html` writes a self-contained interactive HTML file |

In Jupyter notebooks, `chart()` returns a `plotly.graph_objects.Figure` that renders inline automatically.

## Notes

- **Bar vs calendar**: `ma`, `ema`, `bb`, `atr` and all TA-Lib overlays use *bar* periods. `rvol` uses *calendar day* periods (converted internally).
- **Multi-line overlays**: MACD (line + signal + histogram) and Stochastic variants (K + D) render multiple lines on the secondary axis with solid/dashed/dotted styles.
- **Price-axis overlays**: SAR, KAMA, HT Trendline render directly on the price axis.
- **TA-Lib dependency**: Built-in overlays (rvol, ma, ema, bb, vol, atr) work without TA-Lib. TA-Lib overlays raise a clear error with install instructions if TA-Lib is missing.
- **Data**: `candles_1m` from `market.duckdb`, resampled on-the-fly. All 362 Coinbase USD spot pairs available.
- **Discover overlays**: Run `python scripts/chart.py --list` for the full catalog.
