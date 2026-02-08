"""Tests for Phase 6: Observability & Operations.

Covers:
- MetricsCollector: gauges, counters, flush, read
- HTML dashboard: tearsheet generation
- AlertManager: rule evaluation, pre-built rules
- Reconciliation: drift detection, history aggregation
"""
from __future__ import annotations

import json
import shutil
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import polars as pl
import pytest

from monitoring.alerts import (
    Alert,
    AlertManager,
    AlertSeverity,
    concentration_rule,
    max_drawdown_rule,
    risk_limit_rule,
    tracking_error_rule,
)
from monitoring.dashboard import generate_html_tearsheet
from monitoring.metrics import MetricsCollector
from monitoring.reconciliation import (
    DriftReport,
    reconcile_history,
    reconcile_live_vs_target,
)


# ---------------------------------------------------------------------------
# Metrics Collection
# ---------------------------------------------------------------------------


class TestMetricsCollector:
    def setup_method(self) -> None:
        self.tmp_dir = tempfile.mkdtemp()
        self.output_path = Path(self.tmp_dir) / "metrics.jsonl"

    def teardown_method(self) -> None:
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_gauge_records_value(self) -> None:
        mc = MetricsCollector(self.output_path, auto_flush_interval=0)
        mc.gauge("nav", 100_000.0)
        assert mc.get_gauge("nav") == 100_000.0

    def test_counter_increments(self) -> None:
        mc = MetricsCollector(self.output_path, auto_flush_interval=0)
        mc.counter("trades")
        mc.counter("trades")
        mc.counter("trades", increment=3.0)
        assert mc.get_counter("trades") == 5.0

    def test_flush_writes_jsonl(self) -> None:
        mc = MetricsCollector(self.output_path, auto_flush_interval=0)
        mc.gauge("sharpe", 1.5)
        mc.counter("orders", increment=1.0)
        count = mc.flush()
        assert count == 2
        assert self.output_path.exists()
        lines = self.output_path.read_text().strip().split("\n")
        assert len(lines) == 2
        data = json.loads(lines[0])
        assert data["name"] == "sharpe"
        assert data["type"] == "gauge"

    def test_read_all(self) -> None:
        mc = MetricsCollector(self.output_path, auto_flush_interval=0)
        mc.gauge("x", 1.0)
        mc.gauge("y", 2.0)
        mc.flush()
        records = mc.read_all()
        assert len(records) == 2

    def test_auto_flush(self) -> None:
        mc = MetricsCollector(self.output_path, auto_flush_interval=3)
        mc.gauge("a", 1.0)
        mc.gauge("b", 2.0)
        assert not self.output_path.exists()
        mc.gauge("c", 3.0)  # Should trigger auto-flush
        assert self.output_path.exists()

    def test_tags(self) -> None:
        mc = MetricsCollector(self.output_path, auto_flush_interval=0)
        mc.gauge("nav", 100.0, tags={"symbol": "BTC-USD"})
        mc.flush()
        records = mc.read_all()
        assert records[0]["tags"]["symbol"] == "BTC-USD"


# ---------------------------------------------------------------------------
# HTML Dashboard
# ---------------------------------------------------------------------------


class TestDashboard:
    def setup_method(self) -> None:
        self.tmp_dir = tempfile.mkdtemp()

    def teardown_method(self) -> None:
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_generate_html_tearsheet(self) -> None:
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        equity_df = pl.DataFrame({
            "ts": [start + timedelta(hours=i) for i in range(10)],
            "nav": [100.0 + i * 0.5 for i in range(10)],
            "net_ret": [0.005] * 10,
            "dd": [0.0, 0.0, -0.01, -0.02, -0.01, 0.0, 0.0, -0.005, -0.01, 0.0],
        })
        summary = {"total_return": 0.045, "sharpe": 1.5, "max_drawdown": -0.02}
        output = Path(self.tmp_dir) / "test_tearsheet.html"
        result = generate_html_tearsheet(equity_df, summary, output)
        assert result.exists()
        content = result.read_text()
        assert "<html>" in content
        assert "Equity Curve" in content
        assert "Drawdown" in content
        assert "total_return" in content

    def test_empty_equity(self) -> None:
        equity_df = pl.DataFrame({"ts": [], "nav": [], "dd": []})
        summary: dict[str, object] = {}
        output = Path(self.tmp_dir) / "empty.html"
        result = generate_html_tearsheet(equity_df, summary, output)
        assert result.exists()


# ---------------------------------------------------------------------------
# Alerting
# ---------------------------------------------------------------------------


class TestAlertManager:
    def test_alert_fires_on_condition(self) -> None:
        rule = max_drawdown_rule(threshold=0.10)
        manager = AlertManager(rules=[rule])
        alerts = manager.evaluate({"drawdown": -0.15})
        assert len(alerts) == 1
        assert alerts[0].severity == AlertSeverity.CRITICAL

    def test_no_alert_within_threshold(self) -> None:
        rule = max_drawdown_rule(threshold=0.20)
        manager = AlertManager(rules=[rule])
        alerts = manager.evaluate({"drawdown": -0.05})
        assert len(alerts) == 0

    def test_tracking_error_rule(self) -> None:
        rule = tracking_error_rule(threshold=0.03)
        manager = AlertManager(rules=[rule])
        alerts = manager.evaluate({"tracking_error": 0.05})
        assert len(alerts) == 1

    def test_risk_limit_rule(self) -> None:
        rule = risk_limit_rule(max_gross_leverage=1.0)
        manager = AlertManager(rules=[rule])
        alerts = manager.evaluate({"gross_leverage": 1.5})
        assert len(alerts) == 1

    def test_concentration_rule(self) -> None:
        rule = concentration_rule(max_weight=0.3)
        manager = AlertManager(rules=[rule])
        alerts = manager.evaluate({"weights": {"BTC": 0.5, "ETH": 0.2}})
        assert len(alerts) == 1

    def test_callback_invoked(self) -> None:
        fired_alerts: list[Alert] = []
        rule = max_drawdown_rule(threshold=0.01)
        manager = AlertManager(rules=[rule], on_alert=lambda a: fired_alerts.append(a))
        manager.evaluate({"drawdown": -0.05})
        assert len(fired_alerts) == 1

    def test_alert_history(self) -> None:
        rule = max_drawdown_rule(threshold=0.01)
        manager = AlertManager(rules=[rule])
        manager.evaluate({"drawdown": -0.05})
        manager.evaluate({"drawdown": -0.03})
        assert len(manager.alert_history) == 2
        manager.clear_history()
        assert len(manager.alert_history) == 0


# ---------------------------------------------------------------------------
# Reconciliation
# ---------------------------------------------------------------------------


class TestReconciliation:
    def test_clean_reconciliation(self) -> None:
        report = reconcile_live_vs_target(
            expected_weights={"BTC": 0.5, "ETH": 0.3},
            actual_weights={"BTC": 0.51, "ETH": 0.29},
            tolerance=0.02,
        )
        assert isinstance(report, DriftReport)
        assert report.is_clean

    def test_dirty_reconciliation(self) -> None:
        report = reconcile_live_vs_target(
            expected_weights={"BTC": 0.5},
            actual_weights={"BTC": 0.2},
            tolerance=0.05,
        )
        assert not report.is_clean
        assert "BTC" in report.symbols_with_drift
        assert abs(report.drifts["BTC"] - (-0.3)) < 1e-10

    def test_missing_actual_position(self) -> None:
        report = reconcile_live_vs_target(
            expected_weights={"BTC": 0.5, "ETH": 0.3},
            actual_weights={"BTC": 0.5},
            tolerance=0.02,
        )
        assert not report.is_clean
        assert "ETH" in report.symbols_with_drift

    def test_reconcile_history_aggregation(self) -> None:
        reports = [
            reconcile_live_vs_target(
                {"BTC": 0.5}, {"BTC": 0.51}, tolerance=0.02,
            ),
            reconcile_live_vs_target(
                {"BTC": 0.5}, {"BTC": 0.3}, tolerance=0.02,
            ),
            reconcile_live_vs_target(
                {"BTC": 0.5}, {"BTC": 0.49}, tolerance=0.02,
            ),
        ]
        agg = reconcile_history(reports)
        assert agg["n_reports"] == 3
        assert agg["n_clean"] == 2
        assert agg["n_dirty"] == 1
        assert agg["worst_max_drift"] >= 0.19

    def test_empty_history(self) -> None:
        agg = reconcile_history([])
        assert agg["n_reports"] == 0
