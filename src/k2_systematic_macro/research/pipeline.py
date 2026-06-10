"""End-to-end CL research artifact generation."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from k2_systematic_macro.configs.cl import CLResearchConfig
from k2_systematic_macro.csv_export import write_research_csv
from k2_systematic_macro.data.dataset import CanonicalCLDatasetBuilder, ResearchDataset
from k2_systematic_macro.features.targets import TargetConfig, build_target_frame
from k2_systematic_macro.models.expansion import (
    ExpansionModelConfig,
    ExpansionModelResult,
    available_boosters,
    walk_forward_expansion_models,
)
from k2_systematic_macro.regimes.engine import RegimeConfig, fit_regimes
from k2_systematic_macro.regimes.evaluation import RegimeEvaluation, evaluate_regimes
from k2_systematic_macro.research.cl_quality_report import build_cl_quality_report
from k2_systematic_macro.research.signal_report import build_signal_research_artifacts


_PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ARTIFACT_ROOT = _PROJECT_ROOT / "artifacts" / "research" / "k2_systematic_macro"


@dataclass(frozen=True)
class CLResearchPipelineConfig:
    """Controls the local CL diagnostics/modeling artifact run."""

    cl_config: CLResearchConfig = field(default_factory=CLResearchConfig)
    output_root: Path = DEFAULT_ARTIFACT_ROOT
    primary_timeframe: str | None = None
    target_config: TargetConfig = field(default_factory=TargetConfig)
    regime_config: RegimeConfig = field(default_factory=RegimeConfig)
    expansion_config: ExpansionModelConfig = field(default_factory=ExpansionModelConfig)
    include_optional_boosters: bool = True


@dataclass(frozen=True)
class CLResearchPipelineResult:
    """In-memory result plus artifact locations."""

    dataset: ResearchDataset
    primary_panel: pd.DataFrame
    regime_evaluation: RegimeEvaluation
    expansion_results: dict[str, ExpansionModelResult]
    output_dir: Path
    artifact_paths: dict[str, Path]


def run_cl_research_pipeline(
    config: CLResearchPipelineConfig | None = None,
) -> CLResearchPipelineResult:
    """Build canonical CL bars, diagnostics, regimes, labels, and model artifacts."""
    cfg = config or CLResearchPipelineConfig()
    builder = CanonicalCLDatasetBuilder(cfg.cl_config)
    dataset = builder.build(include_features=True)
    dataset_id = dataset.metadata.get("dataset_id", _dataset_id(dataset.symbol))
    output_dir = cfg.output_root / dataset.symbol / str(dataset_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    primary_timeframe = cfg.primary_timeframe or dataset.primary_timeframe
    primary_features = dataset.features[primary_timeframe]
    target_panel = build_target_frame(primary_features, cfg.target_config)
    regime_panel = fit_regimes(target_panel, cfg.regime_config)
    regime_eval = evaluate_regimes(regime_panel)
    model_config = _model_config(cfg.expansion_config, cfg.include_optional_boosters)
    expansion_results = walk_forward_expansion_models(regime_panel, model_config)

    artifact_paths = _write_artifacts(
        output_dir,
        dataset,
        primary_timeframe,
        cfg.cl_config,
        regime_panel,
        regime_eval,
        expansion_results,
        model_config,
        cfg.target_config,
        cfg.regime_config,
        cfg.include_optional_boosters,
    )
    return CLResearchPipelineResult(
        dataset=dataset,
        primary_panel=regime_panel,
        regime_evaluation=regime_eval,
        expansion_results=expansion_results,
        output_dir=output_dir,
        artifact_paths=artifact_paths,
    )


def _model_config(
    config: ExpansionModelConfig,
    include_optional_boosters: bool,
) -> ExpansionModelConfig:
    if not include_optional_boosters:
        return config
    boosters = available_boosters()
    model_kinds = list(config.model_kinds)
    for model_kind, available in boosters.items():
        if available and model_kind not in model_kinds:
            model_kinds.append(model_kind)
    return ExpansionModelConfig(
        horizon=config.horizon,
        target_col=config.target_col,
        model_kinds=tuple(model_kinds),
        min_train_size=config.min_train_size,
        test_size=config.test_size,
        step_size=config.step_size,
        calibration_bins=config.calibration_bins,
        feature_columns=config.feature_columns,
    )


def _write_artifacts(
    output_dir: Path,
    dataset: ResearchDataset,
    primary_timeframe: str,
    cl_config: CLResearchConfig,
    primary_panel: pd.DataFrame,
    regime_eval: RegimeEvaluation,
    expansion_results: dict[str, ExpansionModelResult],
    model_config: ExpansionModelConfig,
    target_config: TargetConfig,
    regime_config: RegimeConfig,
    include_optional_boosters: bool,
) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for timeframe, frame in dataset.bars.items():
        paths[f"bars_{timeframe}"] = _write_frame(frame, output_dir / f"bars_{timeframe}.csv")
    for timeframe, frame in dataset.features.items():
        paths[f"features_{timeframe}"] = _write_frame(frame, output_dir / f"features_{timeframe}.csv")

    paths["primary_panel"] = _write_frame(primary_panel, output_dir / f"panel_{primary_timeframe}.csv")
    paths["regime_transition_matrix"] = _write_frame(
        regime_eval.transition_matrix,
        output_dir / "regime_transition_matrix.csv",
    )
    paths["regime_summary"] = _write_frame(regime_eval.summary, output_dir / "regime_summary.csv")
    paths["regime_persistence"] = _write_frame(regime_eval.persistence, output_dir / "regime_persistence.csv")

    model_metrics: dict[str, Any] = {
        "model_config": {
            "horizon": model_config.horizon,
            "model_kinds": list(model_config.model_kinds),
            "min_train_size": model_config.min_train_size,
            "test_size": model_config.test_size,
            "calibration_bins": model_config.calibration_bins,
        },
        "optional_boosters": available_boosters(),
        "metrics": {},
    }
    for model_kind, result in expansion_results.items():
        model_metrics["metrics"][model_kind] = result.metrics
        paths[f"expansion_predictions_{model_kind}"] = _write_frame(
            result.predictions,
            output_dir / f"expansion_predictions_{model_kind}.csv",
        )
        paths[f"expansion_folds_{model_kind}"] = _write_frame(
            result.fold_metrics,
            output_dir / f"expansion_folds_{model_kind}.csv",
        )
        paths[f"expansion_calibration_{model_kind}"] = _write_frame(
            result.calibration,
            output_dir / f"expansion_calibration_{model_kind}.csv",
        )

    paths["metadata"] = _write_json(output_dir / "metadata.json", dataset.metadata)
    paths["model_metrics"] = _write_json(output_dir / "model_metrics.json", model_metrics)
    signal_artifacts = build_signal_research_artifacts(
        output_dir=output_dir,
        dataset=dataset,
        cl_config=cl_config,
        primary_timeframe=primary_timeframe,
        primary_panel=primary_panel,
        regime_eval=regime_eval,
        expansion_results=expansion_results,
        run_config={
            "target_config": asdict(target_config),
            "regime_config": asdict(regime_config),
            "model_config": asdict(model_config),
            "include_optional_boosters": include_optional_boosters,
        },
    )
    paths.update(signal_artifacts.paths)
    paths["report"] = _write_text(
        output_dir / "report.md",
        _render_report(dataset, primary_timeframe, primary_panel, regime_eval, expansion_results, model_metrics),
    )
    return paths


def _render_report(
    dataset: ResearchDataset,
    primary_timeframe: str,
    primary_panel: pd.DataFrame,
    regime_eval: RegimeEvaluation,
    expansion_results: dict[str, ExpansionModelResult],
    model_metrics: dict[str, Any],
) -> str:
    lines = [
        build_cl_quality_report(dataset).rstrip(),
        "",
        "## Forward Target Coverage",
        "",
        f"- Primary modeling timeframe: `{primary_timeframe}`",
        f"- Rows with assigned regimes: `{int(primary_panel['regime'].notna().sum())}`",
    ]
    for col in sorted(c for c in primary_panel.columns if c.startswith("target_tail_event_")):
        valid = pd.to_numeric(primary_panel[col], errors="coerce").dropna()
        if not valid.empty:
            lines.append(f"- `{col}` event rate: `{valid.mean():.4f}` over `{valid.shape[0]}` rows")
    lines.extend(["", "## Regime Evaluation", ""])
    if regime_eval.summary.empty:
        lines.append("- Regime evaluation was empty after feature filtering.")
    else:
        for row in regime_eval.summary.to_dict(orient="records"):
            lines.append(
                "- Regime `{regime}`: share `{share:.3f}`, vol `{volatility}`, "
                "skew `{skew}`, max drawdown `{max_drawdown}`, tail participation `{tail_participation}`".format(
                    **row
                )
            )
    lines.extend(["", "## Expansion Model Diagnostics", ""])
    lines.append(f"- Optional booster availability: `{model_metrics['optional_boosters']}`")
    for model_kind, result in expansion_results.items():
        lines.append(f"- `{model_kind}` metrics: `{result.metrics}`")
    lines.extend(
        [
            "",
            "## Signal Research Report",
            "",
            "See `signal_research_report.md` and the `graphics/` directory for the "
            "regime-conditioned diagnostics, roll sensitivity, fold diagnostics, and regime-2 overlays.",
            "",
            "## No-Trading Boundary",
            "",
            "These artifacts are diagnostics and supervised-learning baselines only. "
            "They do not define entries, exits, sizing, portfolio construction, or execution rules.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_frame(frame: pd.DataFrame, path: Path) -> Path:
    return write_research_csv(frame, path, index=frame.index.name is not None)


def _write_json(path: Path, payload: Any) -> Path:
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path


def _write_text(path: Path, text: str) -> Path:
    path.write_text(text)
    return path


def _dataset_id(symbol: str) -> str:
    return f"{symbol.upper()}_research_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
