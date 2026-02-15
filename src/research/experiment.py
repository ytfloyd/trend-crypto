"""File-based experiment tracking for backtest research.

Provides a lightweight experiment tracker that logs parameters, metrics,
and artifacts to JSON files. No external dependencies (MLflow, W&B, etc.).
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from common.logging import get_logger

logger = get_logger("experiment")


@dataclass
class ExperimentRun:
    """A single experiment run record.

    Attributes:
        run_id: Unique identifier.
        run_name: Human-readable name.
        params: Hyperparameters and configuration.
        metrics: Computed performance metrics.
        artifacts: Paths to saved artifacts.
        tags: Key-value tags for organization.
        started_at: Start timestamp.
        finished_at: Finish timestamp (None if in progress).
        status: "running", "completed", "failed".
    """

    run_id: str
    run_name: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    artifacts: list[str] = field(default_factory=list)
    tags: dict[str, str] = field(default_factory=dict)
    started_at: str = ""
    finished_at: Optional[str] = None
    status: str = "running"


class ExperimentTracker:
    """File-based experiment tracking.

    Stores runs as individual JSON files in a directory. Supports
    start/log/finish lifecycle and leaderboard queries.

    Args:
        base_dir: Directory to store experiment runs.
    """

    def __init__(self, base_dir: str | Path = "experiments") -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._current_run: Optional[ExperimentRun] = None

    def start_run(
        self,
        run_name: str = "",
        params: Optional[dict[str, Any]] = None,
        tags: Optional[dict[str, str]] = None,
    ) -> ExperimentRun:
        """Start a new experiment run.

        Args:
            run_name: Name for this run.
            params: Initial parameters.
            tags: Key-value tags.

        Returns:
            The new ExperimentRun.
        """
        run = ExperimentRun(
            run_id=str(uuid.uuid4())[:8],
            run_name=run_name,
            params=params or {},
            tags=tags or {},
            started_at=datetime.now(timezone.utc).isoformat(),
            status="running",
        )
        self._current_run = run
        self._save_run(run)
        logger.info("Started experiment run %s: %s", run.run_id, run_name)
        return run

    def log_params(self, params: dict[str, Any]) -> None:
        """Log parameters to the current run."""
        if self._current_run is None:
            raise RuntimeError("No active run. Call start_run() first.")
        self._current_run.params.update(params)
        self._save_run(self._current_run)

    def log_metrics(self, metrics: dict[str, float]) -> None:
        """Log metrics to the current run."""
        if self._current_run is None:
            raise RuntimeError("No active run. Call start_run() first.")
        self._current_run.metrics.update(metrics)
        self._save_run(self._current_run)

    def log_artifact(self, artifact_path: str) -> None:
        """Log an artifact path to the current run."""
        if self._current_run is None:
            raise RuntimeError("No active run. Call start_run() first.")
        self._current_run.artifacts.append(artifact_path)
        self._save_run(self._current_run)

    def finish_run(self, status: str = "completed") -> ExperimentRun:
        """Mark the current run as finished.

        Args:
            status: Final status ("completed" or "failed").

        Returns:
            The finished ExperimentRun.
        """
        if self._current_run is None:
            raise RuntimeError("No active run.")
        self._current_run.status = status
        self._current_run.finished_at = datetime.now(timezone.utc).isoformat()
        self._save_run(self._current_run)
        run = self._current_run
        self._current_run = None
        logger.info("Finished experiment run %s: %s", run.run_id, status)
        return run

    def load_run(self, run_id: str) -> Optional[ExperimentRun]:
        """Load a run by its ID."""
        path = self.base_dir / f"{run_id}.json"
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return ExperimentRun(**data)

    def list_runs(self) -> list[ExperimentRun]:
        """List all experiment runs."""
        runs: list[ExperimentRun] = []
        for path in sorted(self.base_dir.glob("*.json")):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            runs.append(ExperimentRun(**data))
        return runs

    def leaderboard(
        self,
        metric: str = "sharpe",
        ascending: bool = False,
        top_n: int = 10,
    ) -> list[ExperimentRun]:
        """Rank completed runs by a metric.

        Args:
            metric: Metric name to sort by.
            ascending: Sort ascending (True) or descending (False).
            top_n: Number of top runs to return.

        Returns:
            Sorted list of ExperimentRuns.
        """
        runs = [r for r in self.list_runs() if r.status == "completed" and metric in r.metrics]
        runs.sort(key=lambda r: r.metrics.get(metric, 0.0), reverse=not ascending)
        return runs[:top_n]

    def _save_run(self, run: ExperimentRun) -> None:
        """Persist a run to disk as JSON."""
        path = self.base_dir / f"{run.run_id}.json"
        data = {
            "run_id": run.run_id,
            "run_name": run.run_name,
            "params": run.params,
            "metrics": run.metrics,
            "artifacts": run.artifacts,
            "tags": run.tags,
            "started_at": run.started_at,
            "finished_at": run.finished_at,
            "status": run.status,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
