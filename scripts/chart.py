#!/usr/bin/env python
"""Bespoke CLI charting tool for trend_crypto.

Renders interactive Plotly price charts with configurable overlays
from DuckDB market data at any timeframe. Supports zoom, pan,
crosshair hover, and range selectors.

Usage (CLI):
    python scripts/chart.py SOL-USD --tf 4h --overlay rvol:14d
    python scripts/chart.py BTC-USD --overlay ma:20,50 --overlay vol
    python scripts/chart.py ETH-USD --tf 1h --days 30 --overlay bb:20,2 --overlay atr:14
    python scripts/chart.py SOL-USD --tf 4h --overlay rvol:14d --save charts/sol_4h.html

Usage (notebook / Python):
    from chart import chart
    chart("SOL-USD", tf="4h", overlays=["rvol:14d", "ma:20,50"])
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import duckdb
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from volatility.estimators import close_to_close  # noqa: E402

from chart_overlays import compute_talib_overlay, list_overlays  # noqa: E402

# ---------------------------------------------------------------------------
# Palette (matches notebooks/alpha/_setup.py)
# ---------------------------------------------------------------------------
NAVY = "#1B2A4A"
TEAL = "#2E86AB"
RED = "#DC2626"
GOLD = "#D97706"
GREEN = "#059669"
GRAY = "#6B7280"
LIGHT_BG = "#FAFAFA"

OVERLAY_COLORS = [TEAL, RED, GOLD, GREEN, "#7C3AED", "#DB2777"]

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
FREQ_INTERVALS = {
    "5m": "5 minutes",
    "30m": "30 minutes",
    "1h": "1 hour",
    "4h": "4 hours",
    "8h": "8 hours",
    "1d": "1 day",
}

BARS_PER_DAY = {
    "5m": 288.0,
    "30m": 48.0,
    "1h": 24.0,
    "4h": 6.0,
    "8h": 3.0,
    "1d": 1.0,
}


def _resolve_db() -> str:
    env = os.getenv("TREND_CRYPTO_DUCKDB_PATH")
    if env:
        return env
    candidate = _PROJECT_ROOT.parent / "data" / "market.duckdb"
    if candidate.exists():
        return str(candidate)
    candidate2 = _PROJECT_ROOT / "data" / "market.duckdb"
    if candidate2.exists():
        return str(candidate2)
    raise FileNotFoundError(
        "Cannot find market.duckdb. Set TREND_CRYPTO_DUCKDB_PATH or place it "
        "in ../data/market.duckdb relative to the repo root."
    )


def load_symbol_bars(
    symbol: str,
    tf: str = "1d",
    start: str | None = None,
    end: str | None = None,
    db_path: str | None = None,
) -> pd.DataFrame:
    """Load OHLCV bars for a single symbol at a given timeframe."""
    if tf not in FREQ_INTERVALS:
        raise ValueError(f"Unsupported timeframe {tf!r}. Choose from {list(FREQ_INTERVALS)}")

    db_path = db_path or _resolve_db()
    end = end or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    start = start or (datetime.now(timezone.utc) - timedelta(days=180)).strftime("%Y-%m-%d")

    interval = FREQ_INTERVALS[tf]
    con = duckdb.connect(db_path, read_only=True)
    try:
        df = con.execute(
            f"""
            SELECT
                time_bucket(INTERVAL '{interval}', ts) AS ts,
                FIRST(open ORDER BY ts)  AS open,
                MAX(high)                AS high,
                MIN(low)                 AS low,
                LAST(close ORDER BY ts)  AS close,
                SUM(volume)              AS volume
            FROM candles_1m
            WHERE symbol = ?
              AND ts >= CAST(? AS TIMESTAMPTZ)
              AND ts <= CAST(? AS TIMESTAMPTZ)
            GROUP BY time_bucket(INTERVAL '{interval}', ts)
            HAVING FIRST(open ORDER BY ts) > 0
               AND LAST(close ORDER BY ts) > 0
            ORDER BY ts
            """,
            [symbol, start, end],
        ).fetch_df()
    finally:
        con.close()

    if df.empty:
        raise ValueError(f"No data for {symbol} at {tf} between {start} and {end}")

    df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_localize(None)
    return df


# ---------------------------------------------------------------------------
# Overlay parsing and computation
# ---------------------------------------------------------------------------
def parse_overlay(spec: str) -> tuple[str, list[str]]:
    """Parse 'name:param1,param2' into (name, [param1, param2])."""
    if ":" in spec:
        name, params_str = spec.split(":", 1)
        return name, params_str.split(",")
    return spec, []


def _days_to_bars(day_str: str, tf: str) -> int:
    """Convert a day string like '14d' to number of bars."""
    days = int(day_str.rstrip("d"))
    return max(2, int(days * BARS_PER_DAY.get(tf, 1.0)))


def compute_overlay(
    df: pd.DataFrame, name: str, params: list[str], tf: str
) -> dict:
    """Compute an overlay and return rendering metadata.

    Returns dict with keys:
        series: pd.Series or dict of pd.Series
        axis: 'price' | 'secondary' | 'volume'
        label: str
        style: dict of matplotlib kwargs
    """
    close = df["close"]

    if name == "rvol":
        window_str = params[0] if params else "20d"
        window_bars = _days_to_bars(window_str, tf)
        series = close_to_close(close, window=window_bars, ann_factor=365.0)
        return {
            "series": series * 100,
            "axis": "secondary",
            "label": f"RVol {window_str} (ann %)",
            "style": {"linewidth": 1.5, "alpha": 0.85},
        }

    if name == "ma":
        periods = [int(p) for p in params] if params else [20]
        result = {}
        for p in periods:
            result[f"MA {p}"] = close.rolling(p, min_periods=p).mean()
        return {"series": result, "axis": "price", "label": None, "style": {"linewidth": 1.2}}

    if name == "ema":
        periods = [int(p) for p in params] if params else [20]
        result = {}
        for p in periods:
            result[f"EMA {p}"] = close.ewm(span=p, min_periods=p).mean()
        return {"series": result, "axis": "price", "label": None, "style": {"linewidth": 1.2}}

    if name == "bb":
        period = int(params[0]) if params else 20
        k = float(params[1]) if len(params) > 1 else 2.0
        ma = close.rolling(period, min_periods=period).mean()
        std = close.rolling(period, min_periods=period).std()
        return {
            "series": {"upper": ma + k * std, "mid": ma, "lower": ma - k * std},
            "axis": "bollinger",
            "label": f"BB({period}, {k})",
            "style": {},
        }

    if name == "vol":
        return {
            "series": df["volume"],
            "axis": "volume",
            "label": "Volume",
            "style": {},
        }

    if name == "atr":
        period = int(params[0]) if params else 14
        high, low = df["high"], df["low"]
        prev_close = close.shift(1)
        tr = pd.concat(
            [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(period, min_periods=period).mean()
        return {
            "series": atr,
            "axis": "secondary",
            "label": f"ATR({period})",
            "style": {"linewidth": 1.5, "alpha": 0.85},
        }

    result = compute_talib_overlay(df, name, params)
    if result is not None:
        return result

    raise ValueError(
        f"Unknown overlay: {name!r}. Run with --list to see all available overlays."
    )


# ---------------------------------------------------------------------------
# Rendering (Plotly)
# ---------------------------------------------------------------------------
_DASH_CYCLE = ["solid", "dash", "dot", "dashdot"]


def render_chart(
    df: pd.DataFrame,
    overlays: list[dict],
    title: str,
    candles: bool = False,
    save_path: str | None = None,
) -> go.Figure:
    """Render an interactive Plotly chart with all computed overlays."""
    has_volume = any(o["axis"] == "volume" for o in overlays)
    has_secondary = any(o["axis"] in ("secondary", "multi_secondary") for o in overlays)

    specs: list[list[dict]] = []
    row_heights: list[float] = []
    if has_volume:
        specs = [[{"secondary_y": has_secondary}], [{"secondary_y": False}]]
        row_heights = [0.75, 0.25]
    else:
        specs = [[{"secondary_y": has_secondary}]]
        row_heights = [1.0]

    fig = make_subplots(
        rows=len(specs),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=row_heights,
        specs=specs,
    )

    ts = df["ts"]

    if candles:
        fig.add_trace(
            go.Candlestick(
                x=ts,
                open=df["open"],
                high=df["high"],
                low=df["low"],
                close=df["close"],
                increasing_line_color=GREEN,
                decreasing_line_color=RED,
                increasing_fillcolor=GREEN,
                decreasing_fillcolor=RED,
                name="OHLC",
                showlegend=False,
            ),
            row=1,
            col=1,
            secondary_y=False,
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=ts,
                y=df["high"],
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=1,
            col=1,
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=ts,
                y=df["low"],
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor="rgba(27, 42, 74, 0.08)",
                showlegend=False,
                hoverinfo="skip",
            ),
            row=1,
            col=1,
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=ts,
                y=df["close"],
                mode="lines",
                line=dict(color=NAVY, width=1.8),
                name="Close",
            ),
            row=1,
            col=1,
            secondary_y=False,
        )

    color_idx = 0

    for ov in overlays:
        c = OVERLAY_COLORS[color_idx % len(OVERLAY_COLORS)]

        if ov["axis"] == "price":
            series_dict = ov["series"]
            for lbl, s in series_dict.items():
                lc = OVERLAY_COLORS[color_idx % len(OVERLAY_COLORS)]
                fig.add_trace(
                    go.Scatter(
                        x=ts, y=s, mode="lines",
                        line=dict(color=lc, width=1.5),
                        name=lbl,
                    ),
                    row=1, col=1, secondary_y=False,
                )
                color_idx += 1

        elif ov["axis"] == "secondary":
            fig.add_trace(
                go.Scatter(
                    x=ts, y=ov["series"], mode="lines",
                    line=dict(color=c, width=1.5),
                    name=ov["label"],
                    opacity=0.85,
                ),
                row=1, col=1, secondary_y=True,
            )
            color_idx += 1

        elif ov["axis"] == "multi_secondary":
            series_dict = ov["series"]
            for i, (lbl, s) in enumerate(series_dict.items()):
                lc = OVERLAY_COLORS[(color_idx + i) % len(OVERLAY_COLORS)]
                fig.add_trace(
                    go.Scatter(
                        x=ts, y=s, mode="lines",
                        line=dict(color=lc, width=1.5,
                                  dash=_DASH_CYCLE[i % len(_DASH_CYCLE)]),
                        name=lbl,
                        opacity=0.85,
                    ),
                    row=1, col=1, secondary_y=True,
                )
            color_idx += len(series_dict)

        elif ov["axis"] == "bollinger":
            s = ov["series"]
            fig.add_trace(
                go.Scatter(
                    x=ts, y=s["upper"], mode="lines",
                    line=dict(color=TEAL, width=0.8),
                    showlegend=False, hoverinfo="skip",
                ),
                row=1, col=1, secondary_y=False,
            )
            fig.add_trace(
                go.Scatter(
                    x=ts, y=s["lower"], mode="lines",
                    line=dict(color=TEAL, width=0.8),
                    fill="tonexty",
                    fillcolor="rgba(46, 134, 171, 0.12)",
                    name=ov["label"],
                ),
                row=1, col=1, secondary_y=False,
            )
            fig.add_trace(
                go.Scatter(
                    x=ts, y=s["mid"], mode="lines",
                    line=dict(color=TEAL, width=1, dash="dash"),
                    showlegend=False, hoverinfo="skip",
                ),
                row=1, col=1, secondary_y=False,
            )
            color_idx += 1

        elif ov["axis"] == "volume" and has_volume:
            bar_colors = [
                GREEN if df["close"].iloc[i] >= df["open"].iloc[i] else RED
                for i in range(len(df))
            ]
            fig.add_trace(
                go.Bar(
                    x=ts, y=ov["series"],
                    marker_color=bar_colors,
                    opacity=0.5,
                    name="Volume",
                    showlegend=False,
                ),
                row=2, col=1,
            )

    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color=NAVY), x=0.5),
        template="plotly_white",
        hovermode="x unified",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="left", x=0, font=dict(size=11),
        ),
        margin=dict(l=60, r=60, t=80, b=40),
        height=700 if has_volume else 550,
        xaxis_rangeslider_visible=False,
    )

    fig.update_xaxes(
        rangeslider_visible=(not has_volume),
        rangeselector=dict(
            buttons=[
                dict(count=1, label="1M", step="month", stepmode="backward"),
                dict(count=3, label="3M", step="month", stepmode="backward"),
                dict(count=6, label="6M", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1Y", step="year", stepmode="backward"),
                dict(step="all", label="All"),
            ],
            font=dict(size=10),
        ),
        row=1, col=1,
    )

    fig.update_yaxes(title_text="Price", row=1, col=1, secondary_y=False)
    if has_secondary:
        sec_labels = [
            o["label"] for o in overlays
            if o["axis"] in ("secondary", "multi_secondary") and o.get("label")
        ]
        fig.update_yaxes(
            title_text=sec_labels[0] if len(sec_labels) == 1 else "Indicators",
            row=1, col=1, secondary_y=True,
        )
    if has_volume:
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        fig.update_xaxes(rangeslider_visible=True, row=2 if has_volume else 1, col=1)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        if save_path.endswith(".html"):
            fig.write_html(save_path)
        else:
            fig.write_html(save_path + ".html")
            save_path = save_path + ".html"
        print(f"Saved interactive chart to {save_path}")
    else:
        fig.show()

    return fig


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------
def chart(
    symbol: str,
    tf: str = "1d",
    days: int = 180,
    overlays: list[str] | None = None,
    start: str | None = None,
    end: str | None = None,
    candles: bool = False,
    save: str | None = None,
    db: str | None = None,
    title: str | None = None,
) -> go.Figure:
    """Render an interactive chart. Main API for both CLI and notebook use."""
    if start is None:
        start = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
    if end is None:
        end = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    df = load_symbol_bars(symbol, tf=tf, start=start, end=end, db_path=db)

    computed_overlays = []
    for spec in overlays or []:
        name, params = parse_overlay(spec)
        computed_overlays.append(compute_overlay(df, name, params, tf))

    chart_title = title or f"{symbol}  {tf.upper()}  ({start} to {end})"
    return render_chart(df, computed_overlays, chart_title, candles=candles, save_path=save)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bespoke charting tool for trend_crypto market data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  python scripts/chart.py SOL-USD --tf 4h --overlay rvol:14d
  python scripts/chart.py BTC-USD --overlay ma:20,50 --overlay vol
  python scripts/chart.py ETH-USD --tf 1h --days 30 --overlay bb:20,2 --overlay atr:14
  python scripts/chart.py SOL-USD --tf 4h --overlay rvol:14d --save charts/sol_4h.html
""",
    )
    parser.add_argument("symbol", nargs="?", default=None, help="Trading pair (e.g. SOL-USD, BTC-USD)")
    parser.add_argument("--list", action="store_true", dest="list_overlays", help="List all available overlays and exit")
    parser.add_argument("--tf", default="1d", choices=list(FREQ_INTERVALS), help="Bar timeframe (default: 1d)")
    parser.add_argument("--days", type=int, default=180, help="Lookback in calendar days (default: 180)")
    parser.add_argument("--start", default=None, help="Start date (ISO format, overrides --days)")
    parser.add_argument("--end", default=None, help="End date (ISO format, default: today)")
    parser.add_argument("--overlay", action="append", default=[], dest="overlays", help="Overlay spec (repeatable): rvol:14d, ma:20,50, ema:12,26, bb:20,2, vol, atr:14")
    parser.add_argument("--candles", action="store_true", help="Render OHLC candlesticks")
    parser.add_argument("--save", default=None, help="Save interactive HTML chart instead of opening browser")
    parser.add_argument("--db", default=None, help="Override DuckDB path")
    parser.add_argument("--title", default=None, help="Custom chart title")

    args = parser.parse_args()

    if args.list_overlays:
        print("Built-in overlays:")
        print("  rvol:Nd          Rolling realized vol (ann %)")
        print("  ma:N1,N2,...     Simple moving averages")
        print("  ema:N1,N2,...    Exponential moving averages")
        print("  bb:N,K           Bollinger Bands")
        print("  vol              Volume bars")
        print("  atr:N            Average True Range")
        print("\nTA-Lib overlays (requires: pip install TA-Lib):")
        print(list_overlays())
        return

    if args.symbol is None:
        parser.error("symbol is required (or use --list)")

    chart(
        symbol=args.symbol,
        tf=args.tf,
        days=args.days,
        overlays=args.overlays or None,
        start=args.start,
        end=args.end,
        candles=args.candles,
        save=args.save,
        db=args.db,
        title=args.title,
    )


if __name__ == "__main__":
    main()
