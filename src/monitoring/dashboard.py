"""HTML tearsheet/dashboard generation.

Generates a self-contained HTML report with inline SVG charts.
Replaces/extends the PDF tearsheet for browser-friendly output.
Uses only stdlib — no matplotlib dependency required.
"""
from __future__ import annotations

import html
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import polars as pl

from common.logging import get_logger

logger = get_logger("dashboard")


def _svg_line_chart(
    values: list[float],
    labels: list[str],
    width: int = 600,
    height: int = 200,
    title: str = "",
    color: str = "#2563eb",
) -> str:
    """Generate a simple SVG line chart.

    Args:
        values: Y-axis values.
        labels: X-axis labels (same length as values).
        width: Chart width in pixels.
        height: Chart height in pixels.
        title: Chart title.
        color: Line color.

    Returns:
        SVG markup string.
    """
    if not values:
        return ""
    n = len(values)
    min_v = min(values)
    max_v = max(values)
    v_range = max_v - min_v if max_v != min_v else 1.0
    margin_x = 60
    margin_y = 30
    plot_w = width - 2 * margin_x
    plot_h = height - 2 * margin_y

    points = []
    for i, v in enumerate(values):
        x = margin_x + (i / max(1, n - 1)) * plot_w
        y = margin_y + plot_h - ((v - min_v) / v_range) * plot_h
        points.append(f"{x:.1f},{y:.1f}")

    polyline = " ".join(points)
    title_html = html.escape(title)

    return f"""<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
  <text x="{width // 2}" y="16" text-anchor="middle" font-size="14" fill="#333">{title_html}</text>
  <polyline points="{polyline}" fill="none" stroke="{color}" stroke-width="1.5"/>
  <text x="{margin_x}" y="{height - 5}" font-size="10" fill="#666">{html.escape(labels[0] if labels else "")}</text>
  <text x="{width - margin_x}" y="{height - 5}" text-anchor="end" font-size="10" fill="#666">{html.escape(labels[-1] if labels else "")}</text>
  <text x="5" y="{margin_y + 5}" font-size="10" fill="#666">{max_v:.4f}</text>
  <text x="5" y="{height - margin_y}" font-size="10" fill="#666">{min_v:.4f}</text>
</svg>"""


def generate_html_tearsheet(
    equity_df: pl.DataFrame,
    summary: dict[str, object],
    output_path: str | Path = "tearsheet.html",
    title: str = "Backtest Tearsheet",
) -> Path:
    """Generate a self-contained HTML tearsheet.

    Args:
        equity_df: DataFrame with columns: ts, nav, net_ret, dd.
        summary: Summary statistics dict from engine.
        output_path: Path to write the HTML file.
        title: Page title.

    Returns:
        Path to the generated file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Extract chart data
    nav_values = equity_df["nav"].to_list() if "nav" in equity_df.columns else []
    ts_labels = [str(t)[:10] for t in equity_df["ts"].to_list()] if "ts" in equity_df.columns else []
    dd_values = equity_df["dd"].to_list() if "dd" in equity_df.columns else []

    nav_chart = _svg_line_chart(nav_values, ts_labels, title="NAV", color="#2563eb")
    dd_chart = _svg_line_chart(dd_values, ts_labels, title="Drawdown", color="#dc2626")

    # Build summary table
    summary_rows = ""
    for key, value in sorted(summary.items()):
        if isinstance(value, float):
            formatted = f"{value:.6f}"
        else:
            formatted = html.escape(str(value))
        summary_rows += f"<tr><td>{html.escape(str(key))}</td><td>{formatted}</td></tr>\n"

    generated_at = datetime.now(timezone.utc).isoformat()

    html_content = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{html.escape(title)}</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 40px; background: #f8f9fa; }}
h1 {{ color: #1a1a2e; }}
.section {{ background: white; padding: 24px; margin: 16px 0; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
table {{ border-collapse: collapse; width: 100%; }}
td, th {{ border: 1px solid #e5e7eb; padding: 8px 12px; text-align: left; }}
th {{ background: #f3f4f6; }}
tr:nth-child(even) {{ background: #f9fafb; }}
.footer {{ color: #6b7280; font-size: 12px; margin-top: 24px; }}
</style>
</head>
<body>
<h1>{html.escape(title)}</h1>

<div class="section">
<h2>Equity Curve</h2>
{nav_chart}
</div>

<div class="section">
<h2>Drawdown</h2>
{dd_chart}
</div>

<div class="section">
<h2>Summary Statistics</h2>
<table>
<tr><th>Metric</th><th>Value</th></tr>
{summary_rows}
</table>
</div>

<div class="footer">Generated at {generated_at}</div>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    logger.info("HTML tearsheet written to %s", output_path)
    return output_path
