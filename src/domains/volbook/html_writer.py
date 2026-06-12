"""Render a standalone HTML dashboard for the volbook futures tool.

Canvas is convenient for compact artifacts, but the TA-Lib view is a
long, scroll-heavy application. This writer emits a self-contained HTML
file with embedded data and vanilla JS so browser scrolling, native
``details`` disclosures, and table interactions work without canvas host
constraints.
"""
from __future__ import annotations

import json
from pathlib import Path

from .bundle import OhlcvBundle
from .contracts import ASSET_CLASS_BY_ROOT, CATEGORY_ORDER, FuturesSpec


DEFAULT_HTML_PATH = Path("artifacts/volbook/volbook-futures-ohlcv.html")


def _series_payload(bundle: OhlcvBundle) -> list[dict]:
    out: list[dict] = []
    for s in bundle.series:
        try:
            spec = FuturesSpec(
                symbol=s.symbol,
                exchange=s.exchange,
                expiry=s.expiry,
                currency=s.currency,
            )
            label = spec.label
        except Exception:  # noqa: BLE001
            label = f"{s.symbol} {s.expiry}"
        out.append(
            {
                "key": s.key,
                "contract_key": s.contract_key,
                "contract_label": label,
                "category": ASSET_CLASS_BY_ROOT.get(s.symbol, "Other"),
                "category_order": CATEGORY_ORDER,
                "symbol": s.symbol,
                "expiry": s.expiry,
                "exchange": s.exchange,
                "bar_size": s.bar_size,
                "duration": s.duration,
                "what_to_show": s.what_to_show,
                "fetched_at": s.fetched_at,
                "bars": [
                    {"t": b.t, "o": b.o, "h": b.h, "l": b.l, "c": b.c, "v": b.v}
                    for b in s.bars
                ],
                "indicators": s.indicators or {},
                "setups": s.setups or [],
            }
        )
    return out


def render_html(bundle: OhlcvBundle) -> str:
    data = json.dumps(_series_payload(bundle), ensure_ascii=False)
    data = data.replace("</", "<\\/")
    generated = bundle.generated_at or ""
    html = _TEMPLATE.replace("__BUNDLE_JSON__", data)
    html = html.replace("__GENERATED_AT__", generated)
    return html


def write_html(bundle: OhlcvBundle, path: str | Path = DEFAULT_HTML_PATH) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(render_html(bundle), encoding="utf-8")
    return p


_TEMPLATE = r'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Volbook Futures OHLCV</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #0f1117;
      --panel: #151923;
      --panel-2: #1b2130;
      --text: #e5e7eb;
      --muted: #9ca3af;
      --faint: #6b7280;
      --stroke: #303849;
      --up: #30c48d;
      --down: #f36f6f;
      --info: #65a6ff;
      --warn: #f6c453;
      --shadow: rgba(0, 0, 0, 0.24);
    }
    * { box-sizing: border-box; }
    html, body { margin: 0; min-height: 100%; background: var(--bg); color: var(--text); }
    body { font: 13px/1.45 ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    button, select, input { font: inherit; }
    .app { max-width: 1480px; margin: 0 auto; padding: 22px 28px 56px; }
    .topbar {
      position: sticky; top: 0; z-index: 10; margin: -22px -28px 18px; padding: 16px 28px;
      background: color-mix(in srgb, var(--bg) 92%, transparent); backdrop-filter: blur(10px);
      border-bottom: 1px solid var(--stroke);
    }
    .row { display: flex; align-items: center; gap: 12px; flex-wrap: wrap; }
    .between { justify-content: space-between; }
    h1, h2, h3 { margin: 0; line-height: 1.2; }
    h1 { font-size: 24px; }
    h2 { font-size: 18px; margin: 24px 0 10px; }
    h3 { font-size: 14px; }
    .muted { color: var(--muted); }
    .faint { color: var(--faint); }
    .pill {
      display: inline-flex; align-items: center; gap: 4px; min-height: 22px; padding: 2px 8px;
      border: 1px solid var(--stroke); border-radius: 999px; color: var(--muted); background: var(--panel);
      white-space: nowrap;
    }
    .pill.good { color: var(--up); border-color: color-mix(in srgb, var(--up) 36%, var(--stroke)); }
    .pill.bad { color: var(--down); border-color: color-mix(in srgb, var(--down) 36%, var(--stroke)); }
    .pill.info { color: var(--info); border-color: color-mix(in srgb, var(--info) 36%, var(--stroke)); }
    .controls {
      display: grid; grid-template-columns: minmax(220px, 1fr) minmax(150px, 220px) auto auto;
      gap: 12px; align-items: end; margin-top: 14px;
    }
    label { display: grid; gap: 4px; color: var(--muted); font-size: 12px; }
    select, input[type="search"] {
      height: 32px; color: var(--text); background: var(--panel); border: 1px solid var(--stroke);
      border-radius: 6px; padding: 0 9px; min-width: 0;
    }
    .button {
      height: 32px; border: 1px solid var(--stroke); border-radius: 6px; background: var(--panel);
      color: var(--text); padding: 0 10px; cursor: pointer;
    }
    .button:hover, summary:hover { border-color: var(--info); }
    .stats { display: grid; grid-template-columns: repeat(6, minmax(0, 1fr)); gap: 12px; margin: 16px 0; }
    .stat, .card, details {
      background: var(--panel); border: 1px solid var(--stroke); border-radius: 10px;
    }
    .stat { padding: 12px; }
    .stat .value { font-size: 18px; font-weight: 650; }
    .stat .label { color: var(--muted); font-size: 12px; margin-top: 3px; }
    .chart-wrap { background: #101722; border: 1px solid var(--stroke); border-radius: 10px; padding: 10px; }
    svg { display: block; width: 100%; }
    .setup-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(430px, 1fr)); gap: 12px; }
    .card { padding: 14px; }
    .card-title { display: flex; align-items: center; justify-content: space-between; gap: 10px; margin-bottom: 10px; }
    .setup-stats { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 8px; margin: 10px 0; }
    .setup-stats .stat { padding: 9px; background: var(--panel-2); }
    ul { margin: 8px 0 0 18px; padding: 0; }
    li { margin: 3px 0; }
    details { margin: 10px 0; overflow: clip; }
    summary {
      cursor: pointer; padding: 10px 12px; background: var(--panel-2); border-bottom: 1px solid transparent;
      display: flex; justify-content: space-between; align-items: center; gap: 12px;
    }
    details[open] summary { border-bottom-color: var(--stroke); }
    .details-body { padding: 12px; }
    .table-wrap { overflow: auto; max-height: 560px; border: 1px solid var(--stroke); border-radius: 8px; }
    table { width: 100%; border-collapse: collapse; min-width: 760px; }
    th, td { padding: 7px 9px; border-bottom: 1px solid var(--stroke); vertical-align: top; }
    th { position: sticky; top: 0; background: var(--panel-2); color: var(--muted); text-align: left; z-index: 1; }
    td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
    tr:hover td { background: color-mix(in srgb, var(--panel-2) 70%, transparent); }
    tr.clickable { cursor: pointer; }
    .opportunity-controls {
      display: grid; grid-template-columns: repeat(5, minmax(130px, 1fr)); gap: 10px; margin: 10px 0 14px;
    }
    .opportunity-table table { min-width: 1720px; }
    .opportunity-table .rr-col { width: 72px; }
    .opportunity-table .rationale { min-width: 216px; max-width: 216px; white-space: normal; overflow-wrap: anywhere; }
    .section-note { margin: 0 0 10px; color: var(--muted); }
    .search-row { margin: 8px 0 14px; }
    .empty { color: var(--muted); padding: 12px; border: 1px dashed var(--stroke); border-radius: 8px; }
    .footer { margin-top: 28px; color: var(--faint); }
    @media (max-width: 860px) {
      .app { padding: 18px 14px 48px; }
      .topbar { margin: -18px -14px 16px; padding: 14px; }
      .controls, .stats, .setup-stats { grid-template-columns: 1fr 1fr; }
      .setup-grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="app">
    <div class="topbar">
      <div class="row between">
        <div>
          <h1>Volatility Book — Futures OHLCV</h1>
          <div id="subtitle" class="muted"></div>
        </div>
        <div id="pills" class="row"></div>
      </div>
      <div class="controls">
        <label>Contract <select id="contract"></select></label>
        <label>Timeframe <select id="timeframe"></select></label>
        <button id="toIndicators" class="button" type="button">Jump to indicators</button>
        <button id="toBars" class="button" type="button">Jump to recent bars</button>
      </div>
    </div>

    <section id="summary"></section>
    <section id="opportunities"></section>
    <section>
      <div class="row between"><h2>Technical chart</h2><span class="muted">Price/volume with ADX, realized vol, MACD, and RSI subcharts.</span></div>
      <div class="chart-wrap"><svg id="priceChart" viewBox="0 0 1180 760" preserveAspectRatio="none"></svg></div>
    </section>
    <section id="setups"></section>
    <section id="indicators"></section>
    <section id="bars"></section>
    <div class="footer">Generated at __GENERATED_AT__. Self-contained HTML from <code>src/volbook/html_writer.py</code>.</div>
  </div>

  <script>
    const BUNDLE = __BUNDLE_JSON__;
    const GENERATED_AT = "__GENERATED_AT__";
    const $ = (id) => document.getElementById(id);

    function fmtPrice(x) {
      if (!Number.isFinite(x)) return "—";
      const ax = Math.abs(x);
      const digits = ax >= 100 ? 2 : ax >= 10 ? 3 : 4;
      return x.toLocaleString(undefined, { minimumFractionDigits: digits, maximumFractionDigits: digits });
    }
    function fmtNum(x) {
      if (x === null || !Number.isFinite(x)) return "—";
      const ax = Math.abs(x);
      if (ax >= 1e6) return x.toExponential(2);
      if (ax >= 1000) return x.toFixed(2);
      if (ax >= 1) return x.toFixed(4);
      return x.toFixed(6);
    }
    function fmtPct(x) {
      if (!Number.isFinite(x)) return "—";
      return `${x > 0 ? "+" : ""}${(x * 100).toFixed(2)}%`;
    }
    function fmtVol(v) {
      if (!Number.isFinite(v)) return "—";
      if (Math.abs(v) >= 1e6) return `${(v / 1e6).toFixed(2)}M`;
      if (Math.abs(v) >= 1e3) return `${(v / 1e3).toFixed(1)}k`;
      return v.toFixed(0);
    }
    function esc(s) {
      return String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
    }
    function contracts() {
      const seen = new Map();
      for (const s of BUNDLE) {
        if (!seen.has(s.contract_key)) {
          seen.set(s.contract_key, {
            value: s.contract_key,
            label: s.contract_label,
            category: s.category || "Other",
            symbol: s.symbol || "",
            expiry: s.expiry || "",
          });
        }
      }
      const order = ["Interest Rates", "Equities", "Energy", "Metals", "Agriculture", "FX", "Crypto", "Other"];
      return [...seen.values()].sort((a, b) => {
        const ca = order.indexOf(a.category), cb = order.indexOf(b.category);
        return (ca === -1 ? 99 : ca) - (cb === -1 ? 99 : cb) || a.symbol.localeCompare(b.symbol) || a.expiry.localeCompare(b.expiry);
      });
    }
    function timeframes(contractKey) {
      return BUNDLE.filter(s => s.contract_key === contractKey).map(s => s.bar_size);
    }
    function refreshTimeframes() {
      const vals = timeframes($("contract").value);
      $("timeframe").innerHTML = vals.map(v => `<option value="${esc(v)}">${esc(v)}</option>`).join("");
    }
    function currentSeries() {
      const ck = $("contract").value;
      const tf = $("timeframe").value;
      return BUNDLE.find(s => s.contract_key === ck && s.bar_size === tf) || BUNDLE[0];
    }
    function clamp(x, lo = 0, hi = 1) {
      return Math.max(lo, Math.min(hi, x));
    }
    function median(vals) {
      const xs = vals.filter(Number.isFinite).slice().sort((a, b) => a - b);
      if (!xs.length) return NaN;
      const mid = Math.floor(xs.length / 2);
      return xs.length % 2 ? xs[mid] : (xs[mid - 1] + xs[mid]) / 2;
    }
    function curveRankMap() {
      const groups = {};
      for (const s of BUNDLE) {
        const key = `${s.symbol || ""}::${s.bar_size || ""}`;
        (groups[key] ||= new Set()).add(s.expiry || "");
      }
      const ranks = {};
      for (const [key, expiries] of Object.entries(groups)) {
        [...expiries].sort().forEach((expiry, idx) => { ranks[`${key}::${expiry}`] = idx + 1; });
      }
      return ranks;
    }
    function seriesTradability(series, rankMap) {
      const bars = series.bars || [];
      const recent = bars.slice(-20);
      const vols = recent.map(b => Number(b.v) || 0);
      const avgVol20 = vols.length ? vols.reduce((a, b) => a + b, 0) / vols.length : 0;
      const medVol20 = median(vols);
      const latestVol = vols.at(-1) ?? 0;
      const nonzeroRatio = vols.length ? vols.filter(v => v > 0).length / vols.length : 0;
      const staleRatio = bars.length ? bars.filter(b => b.o === b.h && b.h === b.l && b.l === b.c).length / bars.length : 1;
      const curveRank = rankMap[`${series.symbol || ""}::${series.bar_size || ""}::${series.expiry || ""}`] || 99;
      const lower = String(series.bar_size || "").toLowerCase();
      const isHourly = lower.includes("hour");
      const avgThreshold = isHourly ? 2 : 10;
      const medThreshold = isHourly ? 1 : 3;
      const liquidityScore = clamp(
        0.45 * clamp(avgVol20 / avgThreshold) +
        0.25 * clamp((Number.isFinite(medVol20) ? medVol20 : 0) / medThreshold) +
        0.20 * nonzeroRatio +
        0.10 * clamp(latestVol / avgThreshold)
      );
      const dataQualityScore = clamp((1 - staleRatio) * 0.85 + nonzeroRatio * 0.15);
      const curveScore = clamp(1 - Math.max(0, curveRank - 1) * 0.045, 0.35, 1);
      const tradable = liquidityScore >= 0.42 && dataQualityScore >= 0.45 && nonzeroRatio >= (isHourly ? 0.12 : 0.25);
      const notes = [];
      if (!tradable) notes.push("thin");
      if (staleRatio > 0.35) notes.push("stale bars");
      if (curveRank > 5) notes.push(`far curve #${curveRank}`);
      if (nonzeroRatio < 0.5) notes.push("sparse volume");
      return {
        avgVol20,
        medVol20: Number.isFinite(medVol20) ? medVol20 : 0,
        latestVol,
        nonzeroRatio,
        staleRatio,
        curveRank,
        liquidityScore,
        dataQualityScore,
        curveScore,
        tradable,
        notes,
      };
    }
    function allOpportunities() {
      const rows = [];
      const ranks = curveRankMap();
      for (const series of BUNDLE) {
        const tradability = seriesTradability(series, ranks);
        for (const setup of series.setups || []) {
          const score = Number.isFinite(setup.score) ? setup.score : (setup.rr || 0) * (setup.confidence || 0);
          const adjustedScore = score * tradability.liquidityScore * tradability.dataQualityScore * tradability.curveScore;
          if (!liquidityEligible(tradability)) continue;
          const grade = setupGrade(setup);
          rows.push({
            id: `${series.key}::${setup.name || "setup"}::${setup.direction || ""}::${rows.length}`,
            series_key: series.key,
            contract_key: series.contract_key,
            contract_label: series.contract_label,
            category: series.category || "Other",
            symbol: series.symbol,
            expiry: series.expiry,
            exchange: series.exchange,
            bar_size: series.bar_size,
            fetched_at: series.fetched_at || "",
            bars: (series.bars || []).length,
            setup,
            strategy_class: strategyClass(setup),
            strategy_label: strategyLabel(setup),
            setup_grade: grade.label,
            setup_grade_score: grade.score,
            data_quality: dataQualityFlag(series, tradability),
            technical_score: score,
            adjusted_score: adjustedScore,
            tradability,
            conflict: false,
            view_contract_count: 1,
          });
        }
      }
      const collapsed = applyDirectionConflictPenalty(collapseByUnderlyingView(rows));
      collapsed.sort((a, b) =>
        a.strategy_label.localeCompare(b.strategy_label) ||
        (b.setup_grade_score - a.setup_grade_score) ||
        (b.tradability.avgVol20 - a.tradability.avgVol20) ||
        String(a.contract_label).localeCompare(String(b.contract_label))
      );
      return collapsed;
    }
    function uniqueSorted(vals) {
      return [...new Set(vals.filter(Boolean))].sort((a, b) => String(a).localeCompare(String(b)));
    }
    function setupTags(opportunities) {
      return uniqueSorted(opportunities.flatMap(o => o.setup.tags || []));
    }
    function strategyClass(setup) {
      const tags = setup.tags || [];
      if (tags.includes("mean-reversion")) return "mean-reversion";
      if (tags.includes("breakout")) return "breakout";
      if (tags.includes("trend")) return "trend";
      return String(setup.name || "setup").toLowerCase();
    }
    function strategyLabel(setup) {
      const cls = strategyClass(setup);
      if (cls === "trend") return "Trend Continuation";
      if (cls === "breakout") return "Range Breakout";
      if (cls === "mean-reversion") return "Mean Reversion";
      return String(setup.name || "Other");
    }
    function setupGrade(setup) {
      const strength = Number.isFinite(setup.confidence) ? setup.confidence : 0;
      if (strength >= 0.75) return { label: "A", score: 3 };
      if (strength >= 0.60) return { label: "B", score: 2 };
      return { label: "C", score: 1 };
    }
    function dataQualityFlag(series, tradability) {
      const flags = [];
      if (!series || !(series.bars || []).length) flags.push("missing bars");
      if (!Number.isFinite(tradability.avgVol20) || !Number.isFinite(tradability.medVol20)) flags.push("missing volume");
      if (tradability.staleRatio > 0.2) flags.push("stale watch");
      if (tradability.nonzeroRatio < 0.75) flags.push("sparse watch");
      return flags.length ? flags.join(", ") : "ok";
    }
    function liquidityEligible(tradability) {
      return tradability.tradable && tradability.staleRatio <= 0.35 && tradability.nonzeroRatio >= 0.5;
    }
    function collapseByUnderlyingView(rows) {
      const grouped = new Map();
      for (const row of rows) {
        const st = row.setup;
        const key = `${row.symbol || ""}::${st.direction || ""}::${row.strategy_class}::${row.bar_size || ""}`;
        const existing = grouped.get(key);
        const liqRank = (row.tradability.avgVol20 * 1000) + row.tradability.medVol20;
        const existingLiqRank = existing ? (existing.tradability.avgVol20 * 1000) + existing.tradability.medVol20 : -Infinity;
        if (!existing || liqRank > existingLiqRank || (liqRank === existingLiqRank && row.adjusted_score > existing.adjusted_score)) {
          grouped.set(key, { ...row, view_contract_count: (existing?.view_contract_count || 0) + 1 });
        } else {
          existing.view_contract_count = (existing.view_contract_count || 1) + 1;
        }
      }
      return [...grouped.values()];
    }
    function applyDirectionConflictPenalty(rows) {
      const byInstrument = {};
      for (const row of rows) {
        const key = `${row.symbol || ""}::${row.strategy_class}::${row.bar_size || ""}`;
        (byInstrument[key] ||= new Set()).add(row.setup.direction || "");
      }
      return rows.map(row => {
        const key = `${row.symbol || ""}::${row.strategy_class}::${row.bar_size || ""}`;
        const conflicted = byInstrument[key]?.size > 1;
        if (!conflicted) return row;
        return {
          ...row,
          adjusted_score: row.adjusted_score * 0.5,
          setup_grade_score: Math.max(0, row.setup_grade_score - 1),
          conflict: true,
          data_quality: row.data_quality === "ok" ? "direction conflict" : `${row.data_quality}, direction conflict`,
          tradability: {
            ...row.tradability,
            notes: [...row.tradability.notes, "direction conflict penalty"],
          },
        };
      });
    }
    function selectOpportunity(contractKey, barSize) {
      $("contract").value = contractKey;
      refreshTimeframes();
      $("timeframe").value = barSize;
      render();
      $("priceChart").scrollIntoView({ behavior: "smooth", block: "center" });
    }
    function currentOpportunityFilters() {
      return {
        category: $("oppCategory")?.value || "",
        direction: $("oppDirection")?.value || "",
        tag: $("oppTag")?.value || "",
        timeframe: $("oppTimeframe")?.value || "",
        minRr: $("oppMinRr")?.value || "1.0",
        minLiquidity: $("oppMinLiquidity")?.value || "0.0",
        tradableOnly: Boolean($("oppTradableOnly")?.checked),
      };
    }
    function renderOpportunityFilters(opportunities, selected = currentOpportunityFilters()) {
      const categories = uniqueSorted(opportunities.map(o => o.category));
      const directions = uniqueSorted(opportunities.map(o => o.setup.direction));
      const tags = setupTags(opportunities);
      const timeframes = uniqueSorted(opportunities.map(o => o.bar_size));
      const opt = (value, label = value, selectedValue = "") => `<option value="${esc(value)}" ${value === selectedValue ? "selected" : ""}>${esc(label)}</option>`;
      return `
        <div class="opportunity-controls">
          <label>Category <select id="oppCategory">${opt("", "All", selected.category)}${categories.map(v => opt(v, v, selected.category)).join("")}</select></label>
          <label>Direction <select id="oppDirection">${opt("", "All", selected.direction)}${directions.map(v => opt(v, String(v).toUpperCase(), selected.direction)).join("")}</select></label>
          <label>Strategy <select id="oppTag">${opt("", "All", selected.tag)}${tags.map(v => opt(v, v, selected.tag)).join("")}</select></label>
          <label>Timeframe <select id="oppTimeframe">${opt("", "All", selected.timeframe)}${timeframes.map(v => opt(v, v, selected.timeframe)).join("")}</select></label>
          <label>Min RR <input id="oppMinRr" type="search" inputmode="decimal" placeholder="1.0" value="${esc(selected.minRr)}" /></label>
          <label>Min liquidity <input id="oppMinLiquidity" type="search" inputmode="decimal" placeholder="0.0" value="${esc(selected.minLiquidity)}" /></label>
          <label><span>Tradable only</span><input id="oppTradableOnly" type="checkbox" ${selected.tradableOnly ? "checked" : ""} /></label>
        </div>
      `;
    }
    function filteredOpportunities(opportunities) {
      const category = $("oppCategory")?.value || "";
      const direction = $("oppDirection")?.value || "";
      const tag = $("oppTag")?.value || "";
      const timeframe = $("oppTimeframe")?.value || "";
      const minRr = Number.parseFloat($("oppMinRr")?.value || "1.0");
      const minLiquidity = Number.parseFloat($("oppMinLiquidity")?.value || "0.0");
      const tradableOnly = Boolean($("oppTradableOnly")?.checked);
      return opportunities.filter(o => {
        if (category && o.category !== category) return false;
        if (direction && o.setup.direction !== direction) return false;
        if (tag && !(o.setup.tags || []).includes(tag)) return false;
        if (timeframe && o.bar_size !== timeframe) return false;
        if (Number.isFinite(minRr) && (o.setup.rr || 0) < minRr) return false;
        if (Number.isFinite(minLiquidity) && o.tradability.liquidityScore < minLiquidity) return false;
        if (tradableOnly && !o.tradability.tradable) return false;
        return true;
      });
    }
    function opportunityExportRows(opportunities) {
      return opportunities.map((o, i) => {
        const st = o.setup;
        const tq = o.tradability;
        const rationale = (st.rationale || []).join(" · ");
        const notes = tq.notes.length ? tq.notes.join(", ") : "ok for 1-lot screen";
        return [
          i + 1,
          o.strategy_label,
          o.contract_label,
          o.setup_grade,
          o.data_quality,
          fmtNum(st.rr),
          rationale,
          tq.tradable ? "1-lot ok" : "thin/stale",
          o.category,
          o.bar_size,
          o.strategy_class,
          o.view_contract_count,
          o.conflict ? "yes" : "no",
          tq.curveRank,
          st.name,
          String(st.direction || "").toUpperCase(),
          tq.avgVol20,
          tq.medVol20,
          tq.nonzeroRatio,
          tq.staleRatio,
          tq.liquidityScore,
          tq.dataQualityScore,
          tq.curveScore,
          fmtNum(o.technical_score),
          fmtNum(o.adjusted_score),
          st.entry,
          st.stop,
          st.target,
          st.risk,
          st.reward,
          o.fetched_at || "—",
          notes,
        ];
      });
    }
    function crc32Table() {
      const table = [];
      for (let n = 0; n < 256; n++) {
        let c = n;
        for (let k = 0; k < 8; k++) c = (c & 1) ? (0xedb88320 ^ (c >>> 1)) : (c >>> 1);
        table[n] = c >>> 0;
      }
      return table;
    }
    const CRC32_TABLE = crc32Table();
    function crc32(bytes) {
      let c = 0xffffffff;
      for (const b of bytes) c = CRC32_TABLE[(c ^ b) & 0xff] ^ (c >>> 8);
      return (c ^ 0xffffffff) >>> 0;
    }
    function dosDateTime(date = new Date()) {
      const time = (date.getHours() << 11) | (date.getMinutes() << 5) | Math.floor(date.getSeconds() / 2);
      const day = date.getDate();
      const month = date.getMonth() + 1;
      const year = Math.max(1980, date.getFullYear()) - 1980;
      return { date: (year << 9) | (month << 5) | day, time };
    }
    function u16(n) { return [n & 255, (n >>> 8) & 255]; }
    function u32(n) { return [n & 255, (n >>> 8) & 255, (n >>> 16) & 255, (n >>> 24) & 255]; }
    function makeZip(files) {
      const encoder = new TextEncoder();
      const chunks = [];
      const central = [];
      let offset = 0;
      const stamp = dosDateTime();
      for (const file of files) {
        const nameBytes = encoder.encode(file.name);
        const data = encoder.encode(file.content);
        const crc = crc32(data);
        const local = [
          ...u32(0x04034b50), ...u16(20), ...u16(0), ...u16(0), ...u16(stamp.time), ...u16(stamp.date),
          ...u32(crc), ...u32(data.length), ...u32(data.length), ...u16(nameBytes.length), ...u16(0),
          ...nameBytes, ...data
        ];
        chunks.push(new Uint8Array(local));
        central.push({
          nameBytes, crc, size: data.length, offset,
          header: [
            ...u32(0x02014b50), ...u16(20), ...u16(20), ...u16(0), ...u16(0), ...u16(stamp.time), ...u16(stamp.date),
            ...u32(crc), ...u32(data.length), ...u32(data.length), ...u16(nameBytes.length), ...u16(0), ...u16(0),
            ...u16(0), ...u16(0), ...u32(0), ...u32(offset), ...nameBytes
          ]
        });
        offset += local.length;
      }
      const centralOffset = offset;
      for (const c of central) {
        const bytes = new Uint8Array(c.header);
        chunks.push(bytes);
        offset += bytes.length;
      }
      const end = new Uint8Array([
        ...u32(0x06054b50), ...u16(0), ...u16(0), ...u16(central.length), ...u16(central.length),
        ...u32(offset - centralOffset), ...u32(centralOffset), ...u16(0)
      ]);
      chunks.push(end);
      return new Blob(chunks, { type: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" });
    }
    function colName(n) {
      let s = "";
      while (n > 0) {
        const m = (n - 1) % 26;
        s = String.fromCharCode(65 + m) + s;
        n = Math.floor((n - 1) / 26);
      }
      return s;
    }
    function xlsxCell(value, rowIdx, colIdx) {
      const ref = `${colName(colIdx)}${rowIdx}`;
      if (typeof value === "number" && Number.isFinite(value)) return `<c r="${ref}"><v>${value}</v></c>`;
      return `<c r="${ref}" t="inlineStr"><is><t>${esc(value)}</t></is></c>`;
    }
    function xlsxSheet(headers, rows) {
      const allRows = [headers, ...rows];
      const rowXml = allRows.map((row, r) => `<row r="${r + 1}">${row.map((v, c) => xlsxCell(v, r + 1, c + 1)).join("")}</row>`).join("");
      return `<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/></sheetView></sheetViews>
  <cols><col min="1" max="1" width="8" customWidth="1"/><col min="2" max="2" width="18" customWidth="1"/><col min="6" max="6" width="48" customWidth="1"/><col min="27" max="27" width="28" customWidth="1"/></cols>
  <sheetData>${rowXml}</sheetData>
  <pageMargins left="0.4" right="0.4" top="0.5" bottom="0.5" header="0.2" footer="0.2"/>
  <pageSetup orientation="landscape"/>
</worksheet>`;
    }
    function makeXlsx(headers, rows) {
      const files = [
        { name: "[Content_Types].xml", content: `<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/><Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/></Types>` },
        { name: "_rels/.rels", content: `<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>` },
        { name: "xl/workbook.xml", content: `<?xml version="1.0" encoding="UTF-8" standalone="yes"?><workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="Best Opportunities" sheetId="1" r:id="rId1"/></sheets></workbook>` },
        { name: "xl/_rels/workbook.xml.rels", content: `<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/></Relationships>` },
        { name: "xl/worksheets/sheet1.xml", content: xlsxSheet(headers, rows) },
      ];
      return makeZip(files);
    }
    function downloadOpportunitiesExcel() {
      const opportunities = groupedWatchlist(filteredOpportunities(allOpportunities())).flatMap(g => g.rows);
      const headers = [
        "Rank", "Strategy Group", "Contract", "Setup Grade", "Data Quality", "RR", "Rationale", "Tradable",
        "Category", "Timeframe", "Strategy Class", "View Contracts", "Conflict", "Curve #", "Strategy", "Dir", "Avg Vol20", "Med Vol20",
        "Nonzero", "Stale", "Liq", "Data Q Score", "Curve", "Tech Score", "Former Composite", "Entry", "Stop", "Target", "Risk",
        "Reward", "Fetched", "Notes"
      ];
      const rows = opportunityExportRows(opportunities);
      const blob = makeXlsx(headers, rows);
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `volbook-opportunities-${new Date().toISOString().slice(0, 10)}.xlsx`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
    }
    function groupedWatchlist(opportunities) {
      const order = ["Trend Continuation", "Range Breakout", "Mean Reversion"];
      const groups = order.map(label => ({
        label,
        rows: opportunities
          .filter(o => o.strategy_label === label)
          .sort((a, b) =>
            (b.setup_grade_score - a.setup_grade_score) ||
            (b.tradability.avgVol20 - a.tradability.avgVol20) ||
            String(a.contract_label).localeCompare(String(b.contract_label))
          )
          .slice(0, 10),
      }));
      const capped = [];
      let remaining = 30;
      for (const group of groups) {
        const rows = group.rows.slice(0, remaining);
        capped.push({ ...group, rows });
        remaining -= rows.length;
      }
      return capped;
    }
    function renderOpportunityRows(opportunities) {
      const rows = opportunities.map((o, i) => {
        const st = o.setup;
        const tq = o.tradability;
        const directionClass = st.direction === "long" ? "good" : "bad";
        const tradableClass = tq.tradable ? "good" : "bad";
        const rationale = (st.rationale || []).join(" · ");
        const notes = tq.notes.length ? tq.notes.join(", ") : "ok for 1-lot screen";
        return `
          <tr class="clickable" data-contract-key="${esc(o.contract_key)}" data-bar-size="${esc(o.bar_size)}">
            <td class="num">${i + 1}</td>
            <td>${esc(o.contract_label)}</td>
            <td class="num">${esc(o.setup_grade)}</td>
            <td>${esc(o.data_quality)}</td>
            <td class="num rr-col">${fmtNum(st.rr)}</td>
            <td class="rationale">${esc(rationale)}</td>
            <td><span class="pill ${tradableClass}">${tq.tradable ? "1-lot ok" : "thin/stale"}</span></td>
            <td>${esc(o.category)}</td>
            <td>${esc(o.bar_size)}</td>
            <td>${esc(o.strategy_class)}</td>
            <td class="num">${o.view_contract_count || 1}</td>
            <td>${o.conflict ? "yes" : "no"}</td>
            <td class="num">${tq.curveRank}</td>
            <td>${esc(st.name)}</td>
            <td><span class="pill ${directionClass}">${esc(String(st.direction || "").toUpperCase())}</span></td>
            <td class="num">${fmtVol(tq.avgVol20)}</td>
            <td class="num">${fmtVol(tq.medVol20)}</td>
            <td class="num">${(tq.nonzeroRatio * 100).toFixed(0)}%</td>
            <td class="num">${(tq.staleRatio * 100).toFixed(0)}%</td>
            <td class="num">${fmtNum(tq.liquidityScore)}</td>
            <td class="num">${fmtNum(tq.dataQualityScore)}</td>
            <td class="num">${fmtNum(tq.curveScore)}</td>
            <td class="num">${fmtNum(o.technical_score)}</td>
            <td class="num">${fmtNum(o.adjusted_score)}</td>
            <td class="num">${fmtPrice(st.entry)}</td>
            <td class="num">${fmtPrice(st.stop)}</td>
            <td class="num">${fmtPrice(st.target)}</td>
            <td class="num">${fmtPrice(st.risk)}</td>
            <td class="num">${fmtPrice(st.reward)}</td>
            <td>${esc(o.fetched_at || "—")}</td>
            <td>${esc(notes)}</td>
          </tr>
        `;
      }).join("");
      return `
        <div class="table-wrap opportunity-table">
          <table>
            <thead><tr>
              <th class="num">Rank</th><th>Contract</th><th class="num">Setup Grade</th><th>Data Quality</th><th class="num rr-col">RR</th><th class="rationale">Rationale</th>
              <th>Tradable</th><th>Category</th><th>Timeframe</th><th>Strategy Class</th><th class="num">View Contracts</th><th>Conflict</th><th class="num">Curve #</th><th>Strategy</th><th>Dir</th>
              <th class="num">Avg Vol20</th><th class="num">Med Vol20</th><th class="num">Nonzero</th><th class="num">Stale</th>
              <th class="num">Liq</th><th class="num">Data Q Score</th><th class="num">Curve</th><th class="num">Tech Score</th><th class="num">Former Composite</th>
              <th class="num">Entry</th><th class="num">Stop</th><th class="num">Target</th><th class="num">Risk</th><th class="num">Reward</th>
              <th>Fetched</th><th>Notes</th>
            </tr></thead>
            <tbody>${rows}</tbody>
          </table>
        </div>
      `;
    }
    function renderOpportunities() {
      const opportunities = allOpportunities();
      if (!opportunities.length) {
        $("opportunities").innerHTML = `<h2>Discretionary watchlist</h2><div class="empty">No risk/reward setups are available in this bundle. Rerun the refresh with indicators enabled.</div>`;
        return;
      }
      const filtered = filteredOpportunities(opportunities);
      const groups = groupedWatchlist(filtered);
      const visibleRows = groups.reduce((n, g) => n + g.rows.length, 0);
      const longCount = opportunities.filter(o => o.setup.direction === "long").length;
      const shortCount = opportunities.filter(o => o.setup.direction === "short").length;
      const productCount = new Set(opportunities.map(o => o.contract_key)).size;
      const topCategory = opportunities.reduce((acc, o) => {
        acc[o.category] = (acc[o.category] || 0) + 1;
        return acc;
      }, {});
      const topCat = Object.entries(topCategory).sort((a, b) => b[1] - a[1])[0]?.[0] || "—";
      const selected = currentOpportunityFilters();
      $("opportunities").innerHTML = `
        <div class="row between">
          <h2>Discretionary watchlist</h2>
          <div class="row">
            <button id="downloadOpportunities" class="button" type="button">Download Excel</button>
            <span class="muted">Filter-then-group watchlist for human review.</span>
          </div>
        </div>
        <div class="stats">
          <div class="stat"><div class="value">${opportunities.length}</div><div class="label">Valid setups</div></div>
          <div class="stat"><div class="value">${visibleRows}</div><div class="label">Visible cap 30</div></div>
          <div class="stat"><div class="value">${productCount}</div><div class="label">Products with setups</div></div>
          <div class="stat"><div class="value" style="color:var(--up)">${longCount}</div><div class="label">Long setups</div></div>
          <div class="stat"><div class="value" style="color:var(--down)">${shortCount}</div><div class="label">Short setups</div></div>
          <div class="stat"><div class="value">${esc(topCat)}</div><div class="label">Top category</div></div>
        </div>
        ${renderOpportunityFilters(opportunities, selected)}
        <p class="section-note">Filter-then-group output: regime gate, hard liquidity floor, conflict rule, and curve collapse run first; then each strategy class is capped at 10 rows and sorted by Setup Grade, then Avg Vol20. Component scores remain visible for override, but there is no single cross-strategy rank. Click a row to inspect that chart.</p>
        ${visibleRows ? groups.map(g => `
          <h3>${esc(g.label)} <span class="muted">(${g.rows.length}/10)</span></h3>
          ${g.rows.length ? renderOpportunityRows(g.rows) : `<div class="empty">No ${esc(g.label.toLowerCase())} setups after filters.</div>`}
        `).join("") : `<div class="empty">No opportunities match the current filters.</div>`}
      `;
      for (const id of ["oppCategory", "oppDirection", "oppTag", "oppTimeframe", "oppMinRr", "oppMinLiquidity", "oppTradableOnly"]) {
        $(id)?.addEventListener("input", renderOpportunities);
        $(id)?.addEventListener("change", renderOpportunities);
      }
      $("downloadOpportunities")?.addEventListener("click", downloadOpportunitiesExcel);
      document.querySelectorAll("#opportunities tr[data-contract-key]").forEach(tr => {
        tr.addEventListener("click", () => selectOpportunity(tr.dataset.contractKey, tr.dataset.barSize));
      });
    }
    function realizedVol(bars, n, barSize) {
      if (bars.length < 2) return NaN;
      const tail = bars.slice(Math.max(1, bars.length - n));
      const rets = [];
      for (let i = 1; i < tail.length; i++) rets.push(Math.log(tail[i].c / tail[i - 1].c));
      if (!rets.length) return NaN;
      const mean = rets.reduce((a, b) => a + b, 0) / rets.length;
      const variance = rets.reduce((a, r) => a + (r - mean) ** 2, 0) / Math.max(1, rets.length - 1);
      const lower = barSize.toLowerCase();
      const ann = lower.includes("day") ? 252 : lower.includes("hour") ? 252 * 6.5 : 252;
      return Math.sqrt(variance) * Math.sqrt(ann);
    }
    function renderStats(s) {
      const bars = s.bars || [];
      const first = bars[0], last = bars[bars.length - 1];
      const high = Math.max(...bars.map(b => b.h));
      const low = Math.min(...bars.map(b => b.l));
      const change = first && last ? last.c / first.c - 1 : NaN;
      const rv = realizedVol(bars, 20, s.bar_size);
      $("subtitle").textContent = `${s.contract_label} · ${s.exchange} · ${s.bar_size} · ${bars.length} bars · data as of ${s.fetched_at || GENERATED_AT}`;
      const indCount = Object.values(s.indicators || {}).reduce((a, rows) => a + rows.length, 0);
      const catCount = Object.keys(s.indicators || {}).length;
      $("pills").innerHTML = `
        <span class="pill info">${esc(s.what_to_show)}</span>
        <span class="pill good">TA-Lib ${indCount} indicators / ${catCount} groups</span>
        <span class="pill info">${(s.setups || []).length} risk/reward setups</span>
      `;
      $("summary").innerHTML = `
        <div class="stats">
          <div class="stat"><div class="value">${fmtPrice(last?.c)}</div><div class="label">Last</div></div>
          <div class="stat"><div class="value" style="color:${change >= 0 ? "var(--up)" : "var(--down)"}">${fmtPct(change)}</div><div class="label">Period change</div></div>
          <div class="stat"><div class="value">${fmtPrice(first?.o)}</div><div class="label">Period open</div></div>
          <div class="stat"><div class="value">${fmtPrice(high)}</div><div class="label">Period high</div></div>
          <div class="stat"><div class="value">${fmtPrice(low)}</div><div class="label">Period low</div></div>
          <div class="stat"><div class="value">${fmtPct(rv)}</div><div class="label">Realized vol (20-bar)</div></div>
        </div>
      `;
    }
    function renderChart(s) {
      const svg = $("priceChart");
      const bars = s.bars || [];
      if (!bars.length) { svg.innerHTML = ""; return; }
      const W = 1180, L = 58, R = 78;
      const IW = W - L - R;
      const pricePane = { y: 18, h: 360, title: "Price / Volume" };
      const panes = [
        { y: 398, h: 78, title: "ADX 14" },
        { y: 492, h: 78, title: "HV 20 / HV 21" },
        { y: 586, h: 78, title: "MACD 12 26 9" },
        { y: 680, h: 62, title: "RSI 14" },
      ];
      const x = i => L + (i + 0.5) * (IW / bars.length);
      const xs = bars.map((_, i) => x(i));
      const drawFrame = (pane, min, max, ticks = 3) => {
        let out = `<rect x="${L}" y="${pane.y}" width="${IW}" height="${pane.h}" fill="#111a26" stroke="var(--stroke)"/>`;
        for (let i = 0; i <= ticks; i++) {
          const val = min + (i * (max - min)) / ticks;
          const yy = pane.y + ((max - val) / (max - min || 1)) * pane.h;
          out += `<line x1="${L}" x2="${L + IW}" y1="${yy}" y2="${yy}" stroke="var(--stroke)" opacity="0.7"/>`;
          out += `<text x="${L + IW + 8}" y="${yy + 4}" fill="var(--muted)" font-size="10">${fmtPrice(val)}</text>`;
        }
        out += `<text x="${L + 6}" y="${pane.y + 15}" fill="var(--muted)" font-size="11">${esc(pane.title)}</text>`;
        return out;
      };
      const yScale = (pane, min, max, val) => pane.y + ((max - val) / (max - min || 1)) * pane.h;
      const linePath = (vals, pane, min, max) => {
        let d = "";
        vals.forEach((v, i) => {
          if (!Number.isFinite(v)) return;
          const cmd = d ? "L" : "M";
          d += `${cmd}${xs[i].toFixed(1)},${yScale(pane, min, max, v).toFixed(1)}`;
        });
        return d;
      };
      const indicatorSeries = (fn, output = "real") => {
        for (const rows of Object.values(s.indicators || {})) {
          const rec = rows.find(r => r.function === fn);
          if (rec?.outputs?.[output]) return rec.outputs[output];
        }
        return [];
      };
      const align = (arr) => {
        const clean = (arr || []).map(v => Number.isFinite(v) ? v : NaN);
        return Array(Math.max(0, bars.length - clean.length)).fill(NaN).concat(clean).slice(-bars.length);
      };
      const ema = (vals, period) => {
        const out = Array(vals.length).fill(NaN);
        const k = 2 / (period + 1);
        let prev = NaN;
        vals.forEach((v, i) => {
          if (!Number.isFinite(v)) return;
          if (!Number.isFinite(prev)) {
            if (i + 1 >= period) {
              const seed = vals.slice(i + 1 - period, i + 1);
              if (seed.every(Number.isFinite)) prev = seed.reduce((a, b) => a + b, 0) / period;
            }
          } else {
            prev = v * k + prev * (1 - k);
          }
          out[i] = prev;
        });
        return out;
      };
      const fullOrCalculated = (fn, output, calc) => {
        const stored = indicatorSeries(fn, output);
        return stored.length >= bars.length ? align(stored) : calc();
      };
      const calcRSI = (period = 14) => {
        const out = Array(closes.length).fill(NaN);
        if (closes.length <= period) return out;
        let gain = 0, loss = 0;
        for (let i = 1; i <= period; i++) {
          const d = closes[i] - closes[i - 1];
          if (d >= 0) gain += d; else loss -= d;
        }
        let avgGain = gain / period, avgLoss = loss / period;
        out[period] = avgLoss === 0 ? 100 : 100 - 100 / (1 + avgGain / avgLoss);
        for (let i = period + 1; i < closes.length; i++) {
          const d = closes[i] - closes[i - 1];
          avgGain = (avgGain * (period - 1) + Math.max(d, 0)) / period;
          avgLoss = (avgLoss * (period - 1) + Math.max(-d, 0)) / period;
          out[i] = avgLoss === 0 ? 100 : 100 - 100 / (1 + avgGain / avgLoss);
        }
        return out;
      };
      const calcMACD = () => {
        const fast = ema(closes, 12);
        const slow = ema(closes, 26);
        const macd = closes.map((_, i) => Number.isFinite(fast[i]) && Number.isFinite(slow[i]) ? fast[i] - slow[i] : NaN);
        const signal = ema(macd, 9);
        const hist = macd.map((v, i) => Number.isFinite(v) && Number.isFinite(signal[i]) ? v - signal[i] : NaN);
        return { macd, signal, hist };
      };
      const calcADX = (period = 14) => {
        const tr = Array(bars.length).fill(NaN);
        const plusDM = Array(bars.length).fill(0);
        const minusDM = Array(bars.length).fill(0);
        for (let i = 1; i < bars.length; i++) {
          const upMove = bars[i].h - bars[i - 1].h;
          const downMove = bars[i - 1].l - bars[i].l;
          plusDM[i] = upMove > downMove && upMove > 0 ? upMove : 0;
          minusDM[i] = downMove > upMove && downMove > 0 ? downMove : 0;
          tr[i] = Math.max(
            bars[i].h - bars[i].l,
            Math.abs(bars[i].h - bars[i - 1].c),
            Math.abs(bars[i].l - bars[i - 1].c),
          );
        }
        const out = Array(bars.length).fill(NaN);
        if (bars.length <= period * 2) return out;
        let trSm = 0, plusSm = 0, minusSm = 0;
        for (let i = 1; i <= period; i++) {
          trSm += tr[i]; plusSm += plusDM[i]; minusSm += minusDM[i];
        }
        const dx = Array(bars.length).fill(NaN);
        for (let i = period; i < bars.length; i++) {
          if (i > period) {
            trSm = trSm - trSm / period + tr[i];
            plusSm = plusSm - plusSm / period + plusDM[i];
            minusSm = minusSm - minusSm / period + minusDM[i];
          }
          const plusDI = trSm ? 100 * (plusSm / trSm) : 0;
          const minusDI = trSm ? 100 * (minusSm / trSm) : 0;
          dx[i] = plusDI + minusDI ? 100 * Math.abs(plusDI - minusDI) / (plusDI + minusDI) : 0;
        }
        const seed = dx.slice(period, period * 2).filter(Number.isFinite);
        if (seed.length === period) {
          let adx = seed.reduce((a, b) => a + b, 0) / period;
          out[period * 2 - 1] = adx;
          for (let i = period * 2; i < bars.length; i++) {
            adx = ((adx * (period - 1)) + dx[i]) / period;
            out[i] = adx;
          }
        }
        return out;
      };
      const sma = (n) => bars.map((_, i) => {
        if (i + 1 < n) return NaN;
        const w = bars.slice(i + 1 - n, i + 1);
        return w.reduce((a, b) => a + b.c, 0) / n;
      });
      const hv = (n) => bars.map((_, i) => {
        if (i < n) return NaN;
        const rets = [];
        for (let j = i + 1 - n; j <= i; j++) rets.push(Math.log(bars[j].c / bars[j - 1].c));
        const mean = rets.reduce((a, b) => a + b, 0) / rets.length;
        const variance = rets.reduce((a, r) => a + (r - mean) ** 2, 0) / Math.max(1, rets.length - 1);
        return Math.sqrt(variance) * Math.sqrt(s.bar_size.toLowerCase().includes("hour") ? 252 * 6.5 : 252) * 100;
      });
      const closes = bars.map(b => b.c);
      const hi = Math.max(...bars.map(b => b.h));
      const lo = Math.min(...bars.map(b => b.l));
      const pad = (hi - lo) * 0.06 || 1;
      const yMin = lo - pad, yMax = hi + pad;
      const y = p => yScale(pricePane, yMin, yMax, p);
      const bw = Math.max(2, (IW / bars.length) * 0.68);
      let out = drawFrame(pricePane, yMin, yMax, 5);
      bars.forEach((b, i) => {
        const up = b.c >= b.o;
        const color = up ? "var(--up)" : "var(--down)";
        const xx = xs[i];
        const yo = y(b.o), yc = y(b.c), yh = y(b.h), yl = y(b.l);
        out += `<line x1="${xx}" x2="${xx}" y1="${yh}" y2="${yl}" stroke="${color}" stroke-width="1"/>`;
        out += `<rect x="${xx - bw / 2}" y="${Math.min(yo, yc)}" width="${bw}" height="${Math.max(1, Math.abs(yc - yo))}" fill="${color}" opacity="0.9"/>`;
      });
      const maxVol = Math.max(...bars.map(b => b.v || 0), 1);
      bars.forEach((b, i) => {
        const vh = ((b.v || 0) / maxVol) * 66;
        const color = b.c >= b.o ? "var(--up)" : "var(--down)";
        out += `<rect x="${xs[i] - bw / 2}" y="${pricePane.y + pricePane.h - vh}" width="${bw}" height="${vh}" fill="${color}" opacity="0.32"/>`;
      });
      for (const [vals, color] of [[sma(8), "#ffcc66"], [sma(20), "#ed7d31"]]) {
        const d = linePath(vals, pricePane, yMin, yMax);
        if (d) out += `<path d="${d}" fill="none" stroke="${color}" stroke-width="1.2" opacity="0.95"/>`;
      }
      const adx = fullOrCalculated("ADX", "real", () => calcADX(14));
      const adxVals = adx.filter(Number.isFinite);
      out += drawFrame(panes[0], 0, Math.max(50, ...adxVals, 1), 2);
      if (adxVals.length) out += `<path d="${linePath(adx, panes[0], 0, Math.max(50, ...adxVals, 1))}" fill="none" stroke="#e86d75" stroke-width="1.3"/>`;
      const hv20 = hv(20), hv21 = hv(21);
      const hvMax = Math.max(40, ...hv20.filter(Number.isFinite), ...hv21.filter(Number.isFinite));
      out += drawFrame(panes[1], 0, hvMax, 2);
      out += `<path d="${linePath(hv20, panes[1], 0, hvMax)}" fill="none" stroke="#37b6ff" stroke-width="1.2"/>`;
      out += `<path d="${linePath(hv21, panes[1], 0, hvMax)}" fill="none" stroke="#7bdff2" stroke-width="1.1" opacity="0.75"/>`;
      const macdCalc = calcMACD();
      const macd = fullOrCalculated("MACD", "macd", () => macdCalc.macd);
      const macdsignal = fullOrCalculated("MACD", "macdsignal", () => macdCalc.signal);
      const macdhist = fullOrCalculated("MACD", "macdhist", () => macdCalc.hist);
      const macdVals = [...macd, ...macdsignal, ...macdhist].filter(Number.isFinite);
      const mm = Math.max(1e-9, ...macdVals.map(v => Math.abs(v)));
      out += drawFrame(panes[2], -mm, mm, 2);
      macdhist.forEach((v, i) => {
        if (!Number.isFinite(v)) return;
        const zero = yScale(panes[2], -mm, mm, 0), yy = yScale(panes[2], -mm, mm, v);
        out += `<rect x="${xs[i] - bw / 2}" y="${Math.min(zero, yy)}" width="${bw}" height="${Math.max(1, Math.abs(zero - yy))}" fill="${v >= 0 ? "var(--up)" : "var(--down)"}" opacity="0.7"/>`;
      });
      out += `<path d="${linePath(macd, panes[2], -mm, mm)}" fill="none" stroke="#37b6ff" stroke-width="1.1"/>`;
      out += `<path d="${linePath(macdsignal, panes[2], -mm, mm)}" fill="none" stroke="#ed7d31" stroke-width="1.1"/>`;
      const rsi = fullOrCalculated("RSI", "real", () => calcRSI(14));
      out += drawFrame(panes[3], 0, 100, 2);
      out += `<rect x="${L}" y="${yScale(panes[3], 0, 100, 70)}" width="${IW}" height="${yScale(panes[3], 0, 100, 30) - yScale(panes[3], 0, 100, 70)}" fill="#6d4bb8" opacity="0.18"/>`;
      out += `<path d="${linePath(rsi, panes[3], 0, 100)}" fill="none" stroke="#a78bfa" stroke-width="1.2"/>`;
      const last = bars.at(-1);
      out += `<text x="${L + 8}" y="${pricePane.y + 32}" fill="var(--text)" font-size="12">O ${fmtPrice(last.o)}  H ${fmtPrice(last.h)}  L ${fmtPrice(last.l)}  C ${fmtPrice(last.c)}</text>`;
      svg.innerHTML = out;
    }
    function renderSetups(s) {
      const setups = s.setups || [];
      if (!setups.length) {
        $("setups").innerHTML = `<h2>Selected contract setups</h2><div class="empty">No actionable trend, mean-reversion, or breakout setups fired on the latest bar.</div>`;
        return;
      }
      const cards = setups.map((st, idx) => `
        <div class="card">
          <div class="card-title"><h3>${idx === 0 ? "#1 " : ""}${esc(st.name)} · ${esc(st.direction).toUpperCase()}</h3><span class="pill ${st.direction === "long" ? "good" : "bad"}">RR ${st.rr.toFixed(2)} · conf ${(st.confidence * 100).toFixed(0)}%</span></div>
          <div class="setup-stats">
            <div class="stat"><div class="value">${fmtPrice(st.entry)}</div><div class="label">Entry</div></div>
            <div class="stat"><div class="value" style="color:var(--down)">${fmtPrice(st.stop)}</div><div class="label">Stop</div></div>
            <div class="stat"><div class="value" style="color:${st.direction === "long" ? "var(--up)" : "var(--down)"}">${fmtPrice(st.target)}</div><div class="label">Target</div></div>
            <div class="stat"><div class="value">${(st.confidence * 100).toFixed(0)}%</div><div class="label">Confidence</div></div>
          </div>
          <div class="muted">Risk ${fmtPrice(st.risk)} · Reward ${fmtPrice(st.reward)} · tags ${(st.tags || []).map(esc).join(", ")}</div>
          <ul>${(st.rationale || []).map(r => `<li>${esc(r)}</li>`).join("")}</ul>
        </div>`).join("");
      $("setups").innerHTML = `<h2>Selected contract setups</h2><p class="section-note">Local setups for the selected instrument, ranked by RR x confidence. Stops and targets are anchored to ATR/Bollinger/range levels.</p><div class="setup-grid">${cards}</div>`;
    }
    function renderIndicators(s) {
      const indicators = s.indicators || {};
      const cats = Object.keys(indicators);
      const total = Object.values(indicators).reduce((a, rows) => a + rows.length, 0);
      if (!cats.length) {
        $("indicators").innerHTML = `<h2>Technical indicators</h2><div class="empty">No TA-Lib indicators attached.</div>`;
        return;
      }
      const patternRows = indicators["Pattern Recognition"] || [];
      const fires = [];
      for (const rec of patternRows) {
        const arr = rec.outputs?.integer || [];
        for (let j = arr.length - 1; j >= 0; j--) {
          const v = arr[j];
          if (v && Number.isFinite(v)) fires.push({ rec, offset: arr.length - 1 - j, v });
        }
      }
      fires.sort((a, b) => a.offset - b.offset || a.rec.function.localeCompare(b.rec.function));
      const patternHtml = patternRows.length ? `
        <details open>
          <summary><strong>Pattern Recognition — recent firings</strong><span class="pill info">${fires.length} recent</span></summary>
          <div class="details-body">${fires.length ? table(
            ["Bars ago", "Pattern", "Function", "Signal"],
            fires.map(f => [f.offset === 0 ? "now" : String(f.offset), f.rec.display_name, f.rec.function, f.v > 0 ? `Bullish (${f.v})` : `Bearish (${f.v})`])
          ) : `<div class="empty">No candlestick patterns fired in the stored indicator tail.</div>`}</div>
        </details>` : "";
      const sections = cats.filter(c => c !== "Pattern Recognition").map(cat => {
        const rows = [];
        for (const rec of indicators[cat]) {
          const outs = Object.keys(rec.outputs || {});
          if (!outs.length) {
            rows.push([rec.function, params(rec.params), "—", rec.error ? `error: ${rec.error}` : "—", "—", "—"]);
          } else {
            outs.forEach((name, idx) => {
              const arr = rec.outputs[name] || [];
              const curr = arr.at(-1) ?? null;
              const prev = arr.at(-2) ?? null;
              rows.push([idx === 0 ? rec.function : "", idx === 0 ? params(rec.params) : "", name, fmtNum(curr), fmtNum(prev), change(curr, prev)]);
            });
          }
        }
        return `<details><summary><strong>${esc(cat)}</strong><span class="pill">${indicators[cat].length}</span></summary><div class="details-body">${table(["Function", "Params", "Output", "Latest", "Prev", "Change"], rows, [3,4,5])}</div></details>`;
      }).join("");
      $("indicators").innerHTML = `
        <h2>Technical indicators</h2>
        <p class="section-note">${total} TA-Lib indicators across ${cats.length} categories. Expand groups below; headers stay sticky inside each table.</p>
        <div class="search-row"><label>Filter visible rows <input id="indicatorSearch" type="search" placeholder="RSI, MACD, ATR, BBANDS..." /></label></div>
        <div id="indicatorGroups">${patternHtml}${sections}</div>
      `;
      $("indicatorSearch").addEventListener("input", (e) => {
        const q = e.target.value.toLowerCase().trim();
        document.querySelectorAll("#indicatorGroups tbody tr").forEach(tr => {
          tr.style.display = !q || tr.textContent.toLowerCase().includes(q) ? "" : "none";
        });
      });
    }
    function params(obj) {
      const entries = Object.entries(obj || {});
      return entries.length ? entries.map(([k, v]) => `${k}=${v}`).join(", ") : "—";
    }
    function change(curr, prev) {
      if (curr === null || prev === null || !Number.isFinite(curr) || !Number.isFinite(prev)) return "—";
      const d = curr - prev;
      return `${d > 0 ? "+" : ""}${fmtNum(d)}`;
    }
    function table(headers, rows, numCols = []) {
      return `<div class="table-wrap"><table><thead><tr>${headers.map((h, i) => `<th class="${numCols.includes(i) ? "num" : ""}">${esc(h)}</th>`).join("")}</tr></thead><tbody>${rows.map(r => `<tr>${r.map((c, i) => `<td class="${numCols.includes(i) ? "num" : ""}">${esc(c)}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`;
    }
    function renderBars(s) {
      const rows = (s.bars || []).slice(-60).reverse().map(b => [b.t, fmtPrice(b.o), fmtPrice(b.h), fmtPrice(b.l), fmtPrice(b.c), fmtVol(b.v)]);
      $("bars").innerHTML = `<h2>Recent bars</h2>${table(["Date", "Open", "High", "Low", "Close", "Volume"], rows, [1,2,3,4,5])}`;
    }
    function render() {
      const s = currentSeries();
      renderStats(s);
      renderOpportunities();
      renderChart(s);
      renderSetups(s);
      renderIndicators(s);
      renderBars(s);
    }
    function init() {
      if (!BUNDLE.length) {
        document.body.innerHTML = "<div class='app'><h1>Volbook Futures OHLCV</h1><div class='empty'>No series available. Run the volbook CLI first.</div></div>";
        return;
      }
      const contractSelect = $("contract");
      const grouped = contracts().reduce((acc, c) => {
        (acc[c.category] ||= []).push(c);
        return acc;
      }, {});
      contractSelect.innerHTML = Object.entries(grouped).map(([category, rows]) => `
        <optgroup label="${esc(category)}">
          ${rows.map(c => `<option value="${esc(c.value)}">${esc(c.label)}</option>`).join("")}
        </optgroup>
      `).join("");
      contractSelect.addEventListener("change", () => { refreshTimeframes(); render(); });
      $("timeframe").addEventListener("change", render);
      $("toIndicators").addEventListener("click", () => $("indicators").scrollIntoView({ behavior: "smooth", block: "start" }));
      $("toBars").addEventListener("click", () => $("bars").scrollIntoView({ behavior: "smooth", block: "start" }));
      refreshTimeframes();
      render();
    }
    init();
  </script>
</body>
</html>
'''
