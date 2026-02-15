"""Alerting framework for live trading and risk monitoring.

Provides rule-based alerting with pre-built rules for common risk events.
Alerts are logged and can optionally trigger callbacks.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Optional

from common.logging import get_logger

logger = get_logger("alerts")


class AlertSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass(frozen=True)
class AlertRule:
    """Definition of an alert rule.

    Attributes:
        name: Rule name.
        description: Human-readable description.
        severity: Alert severity level.
        check_fn: Function that takes current state and returns True if alert should fire.
    """

    name: str
    description: str
    severity: AlertSeverity
    check_fn: Callable[[dict[str, Any]], bool]


@dataclass(frozen=True)
class Alert:
    """A fired alert instance.

    Attributes:
        rule_name: Name of the rule that fired.
        severity: Severity level.
        message: Alert message.
        ts: Timestamp when the alert fired.
        context: Additional context data.
    """

    rule_name: str
    severity: AlertSeverity
    message: str
    ts: str
    context: dict[str, Any] = field(default_factory=dict)


class AlertManager:
    """Manages alert rules and fires alerts based on current state.

    Args:
        rules: List of alert rules to evaluate.
        on_alert: Optional callback invoked when an alert fires.
    """

    def __init__(
        self,
        rules: Optional[list[AlertRule]] = None,
        on_alert: Optional[Callable[[Alert], None]] = None,
    ) -> None:
        self.rules = list(rules) if rules else []
        self.on_alert = on_alert
        self._alert_history: list[Alert] = []

    def add_rule(self, rule: AlertRule) -> None:
        """Add an alert rule."""
        self.rules.append(rule)

    def evaluate(self, state: dict[str, Any]) -> list[Alert]:
        """Evaluate all rules against the current state.

        Args:
            state: Dict of current metrics/state values.

        Returns:
            List of alerts that fired.
        """
        fired: list[Alert] = []
        for rule in self.rules:
            try:
                if rule.check_fn(state):
                    alert = Alert(
                        rule_name=rule.name,
                        severity=rule.severity,
                        message=rule.description,
                        ts=datetime.now(timezone.utc).isoformat(),
                        context=dict(state),
                    )
                    fired.append(alert)
                    self._alert_history.append(alert)
                    logger.warning(
                        "ALERT [%s] %s: %s",
                        rule.severity.value, rule.name, rule.description,
                    )
                    if self.on_alert:
                        self.on_alert(alert)
            except Exception as e:
                logger.error("Alert rule '%s' failed: %s", rule.name, e)
        return fired

    @property
    def alert_history(self) -> list[Alert]:
        return list(self._alert_history)

    def clear_history(self) -> None:
        self._alert_history.clear()


# ---------------------------------------------------------------------------
# Pre-built alert rules
# ---------------------------------------------------------------------------


def max_drawdown_rule(threshold: float = 0.20) -> AlertRule:
    """Alert when drawdown exceeds threshold.

    Args:
        threshold: Maximum drawdown (positive number, e.g. 0.20 = 20%).
    """
    return AlertRule(
        name="max_drawdown_breach",
        description=f"Drawdown exceeds {threshold:.0%}",
        severity=AlertSeverity.CRITICAL,
        check_fn=lambda state: abs(state.get("drawdown", 0.0)) >= threshold,
    )


def tracking_error_rule(threshold: float = 0.05) -> AlertRule:
    """Alert when tracking error exceeds threshold."""
    return AlertRule(
        name="tracking_error",
        description=f"Tracking error exceeds {threshold:.2%}",
        severity=AlertSeverity.WARNING,
        check_fn=lambda state: abs(state.get("tracking_error", 0.0)) >= threshold,
    )


def risk_limit_rule(max_gross_leverage: float = 1.0) -> AlertRule:
    """Alert when gross leverage exceeds limit."""
    return AlertRule(
        name="risk_limit_violation",
        description=f"Gross leverage exceeds {max_gross_leverage:.1f}x",
        severity=AlertSeverity.CRITICAL,
        check_fn=lambda state: state.get("gross_leverage", 0.0) > max_gross_leverage,
    )


def concentration_rule(max_weight: float = 0.5) -> AlertRule:
    """Alert when any single position exceeds concentration limit."""
    return AlertRule(
        name="concentration_breach",
        description=f"Single position exceeds {max_weight:.0%}",
        severity=AlertSeverity.WARNING,
        check_fn=lambda state: any(
            abs(w) > max_weight
            for w in state.get("weights", {}).values()
        ),
    )
