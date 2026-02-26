"""
Catalogue manager for Alpha Lab results.

Handles:
  - Incremental JSON persistence (append-safe, resumable)
  - Ranking and filtering of results
  - Markdown summary generation
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .harness import HarnessResult


class Catalogue:
    """Persistent, resumable catalogue of alpha signal test results."""

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._results_path = self.output_dir / "results.jsonl"
        self._summary_path = self.output_dir / "summary.md"
        self._tested: set[str] = set()
        self._results: list[dict] = []
        self._load_existing()

    def _load_existing(self) -> None:
        """Load previously saved results for resumability."""
        if not self._results_path.exists():
            return
        with open(self._results_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    self._results.append(rec)
                    self._tested.add(rec["name"])
                except (json.JSONDecodeError, KeyError):
                    continue
        if self._tested:
            print(f"[catalogue] Resumed with {len(self._tested)} previously tested signals")

    def already_tested(self, name: str) -> bool:
        return name in self._tested

    def record(self, result: HarnessResult) -> None:
        """Append a single result to the catalogue."""
        rec = result.to_dict()
        rec["tested_at"] = datetime.now(timezone.utc).isoformat()
        self._results.append(rec)
        self._tested.add(result.spec.name)
        with open(self._results_path, "a") as f:
            f.write(json.dumps(rec, default=_json_default) + "\n")

    def get_ranked(
        self,
        metric: str = "sharpe",
        mode: str = "long_short",
        min_sharpe: float = -999,
    ) -> list[dict]:
        """Return results sorted by a metric, best first."""
        scored = []
        for r in self._results:
            m = r.get(mode, {})
            if not m or r.get("error"):
                continue
            val = m.get(metric, np.nan)
            if np.isnan(val) or val < min_sharpe:
                continue
            scored.append(r)
        scored.sort(key=lambda x: x.get(mode, {}).get(metric, -999), reverse=True)
        return scored

    def write_summary(self) -> Path:
        """Generate a ranked markdown summary of all results."""
        lines: list[str] = []
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        lines.append(f"# Alpha Lab — Signal Catalogue")
        lines.append(f"")
        lines.append(f"**Generated**: {now}")
        lines.append(f"**Signals tested**: {len(self._results)}")

        n_errors = sum(1 for r in self._results if r.get("error"))
        n_success = len(self._results) - n_errors
        lines.append(f"**Successful**: {n_success}  |  **Errors**: {n_errors}")
        lines.append("")

        # --- Top performers: Long-Short ---
        lines.append("## Top 30 — Long-Short (by Sharpe)")
        lines.append("")
        ranked_ls = self.get_ranked(metric="sharpe", mode="long_short")[:30]
        if ranked_ls:
            lines.append(_markdown_table(ranked_ls, "long_short"))
        else:
            lines.append("*No valid long-short results.*")
        lines.append("")

        # --- Top performers: Long-Only ---
        lines.append("## Top 30 — Long-Only (by Sharpe)")
        lines.append("")
        ranked_lo = self.get_ranked(metric="sharpe", mode="long_only")[:30]
        if ranked_lo:
            lines.append(_markdown_table(ranked_lo, "long_only"))
        else:
            lines.append("*No valid long-only results.*")
        lines.append("")

        # --- Best by family ---
        lines.append("## Best Signal per Family (Long-Short)")
        lines.append("")
        families: dict[str, dict] = {}
        for r in self.get_ranked(metric="sharpe", mode="long_short"):
            fam = r.get("family", "unknown")
            if fam not in families:
                families[fam] = r
        if families:
            lines.append(_markdown_table(list(families.values()), "long_short"))
        lines.append("")

        # --- Regime analysis for top 10 ---
        lines.append("## Regime Breakdown — Top 10 Long-Short")
        lines.append("")
        top10 = self.get_ranked(metric="sharpe", mode="long_short")[:10]
        if top10:
            lines.append(
                "| Signal | BULL | BEAR | CHOP | Complement? |"
            )
            lines.append("|--------|------|------|------|-------------|")
            for r in top10:
                reg = r.get("regime", {})
                bull = reg.get("BULL", {}).get("sharpe", np.nan)
                bear = reg.get("BEAR", {}).get("sharpe", np.nan)
                chop = reg.get("CHOP", {}).get("sharpe", np.nan)
                complement = ""
                if not np.isnan(bull) and not np.isnan(bear):
                    if bull > 0.3 and bear < -0.3:
                        complement = "BULL-only"
                    elif bear > 0.3 and bull < -0.3:
                        complement = "BEAR-only"
                    elif bull > 0.2 and bear > 0.2:
                        complement = "All-weather"
                lines.append(
                    f"| {r['name']:<28s} | {_fmt(bull):>4s} | {_fmt(bear):>4s} "
                    f"| {_fmt(chop):>4s} | {complement} |"
                )
        lines.append("")

        # --- IC summary for top 10 ---
        lines.append("## Information Coefficient — Top 10 Long-Short")
        lines.append("")
        if top10:
            lines.append("| Signal | IC 1d | IC 5d | t-stat 1d |")
            lines.append("|--------|-------|-------|-----------|")
            for r in top10:
                ic = r.get("ic", {})
                ic1 = ic.get("1d", {})
                ic5 = ic.get("5d", {})
                lines.append(
                    f"| {r['name']:<28s} "
                    f"| {_fmt(ic1.get('ic_mean'), 4):>5s} "
                    f"| {_fmt(ic5.get('ic_mean'), 4):>5s} "
                    f"| {_fmt(ic1.get('ic_tstat'), 2):>9s} |"
                )
        lines.append("")

        # --- Error log ---
        errors = [r for r in self._results if r.get("error")]
        if errors:
            lines.append("## Errors")
            lines.append("")
            for r in errors[:20]:
                err_short = r["error"].strip().split("\n")[-1][:120]
                lines.append(f"- **{r['name']}**: `{err_short}`")
            if len(errors) > 20:
                lines.append(f"- ... and {len(errors) - 20} more")
            lines.append("")

        text = "\n".join(lines)
        self._summary_path.write_text(text)
        return self._summary_path

    @property
    def n_tested(self) -> int:
        return len(self._results)

    @property
    def n_remaining(self) -> int:
        return 0  # caller tracks this


def _markdown_table(results: list[dict], mode: str) -> str:
    """Build a markdown table from ranked results."""
    lines = [
        "| # | Signal | Family | Sharpe | CAGR | MaxDD | Sortino | Calmar | Hit% | Turnover |",
        "|---|--------|--------|--------|------|-------|---------|--------|------|----------|",
    ]
    for i, r in enumerate(results, 1):
        m = r.get(mode, {})
        lines.append(
            f"| {i} "
            f"| {r['name']:<28s} "
            f"| {r.get('family', ''):<16s} "
            f"| {_fmt(m.get('sharpe'), 2):>6s} "
            f"| {_pct(m.get('cagr')):>5s} "
            f"| {_pct(m.get('max_dd')):>6s} "
            f"| {_fmt(m.get('sortino'), 2):>7s} "
            f"| {_fmt(m.get('calmar'), 2):>6s} "
            f"| {_pct(m.get('hit_rate')):>4s} "
            f"| {_fmt(m.get('avg_turnover'), 4):>8s} |"
        )
    return "\n".join(lines)


def _to_real(val: Any) -> Any:
    """Convert complex or numpy types to plain float; pass through None/NaN."""
    if val is None:
        return None
    if isinstance(val, (complex, np.complexfloating)):
        val = float(np.real(val))
    if isinstance(val, (np.floating, np.integer)):
        val = float(val)
    if isinstance(val, float) and (np.isnan(val) or np.isinf(val)):
        return None
    return val


def _fmt(val: Any, decimals: int = 2) -> str:
    val = _to_real(val)
    if val is None:
        return "—"
    return f"{val:.{decimals}f}"


def _pct(val: Any) -> str:
    val = _to_real(val)
    if val is None:
        return "—"
    return f"{val:.1%}"


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        if np.isnan(v) or np.isinf(v):
            return None
        return v
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, complex):
        return float(obj.real)
    if isinstance(obj, np.complexfloating):
        return float(obj.real)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
