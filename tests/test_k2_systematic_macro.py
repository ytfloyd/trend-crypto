"""Tests for the K2 systematic macro first research slice."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from k2_systematic_macro.configs.cl import CLResearchConfig
from k2_systematic_macro.csv_export import write_research_csv
from k2_systematic_macro.data.dataset import CanonicalCLDatasetBuilder
from k2_systematic_macro.data.volbook_adapter import VolbookLoadSpec, VolbookResearchDataAdapter
from k2_systematic_macro.features.core import FeatureConfig, build_feature_frame
from k2_systematic_macro.features.targets import TargetConfig, build_target_frame
from k2_systematic_macro.models.expansion import ExpansionModelConfig, walk_forward_expansion_models
from k2_systematic_macro.regimes.engine import RegimeConfig, fit_regimes
from k2_systematic_macro.regimes.evaluation import evaluate_regimes
from k2_systematic_macro.research.cl_quality_report import build_cl_quality_report
from k2_systematic_macro.research.pipeline import CLResearchPipelineConfig, run_cl_research_pipeline
from k2_systematic_macro.research.signal_report import (
    DiagnosticsThresholdConfig,
    TopCellSelectorConfig,
    _convexity_research_score,
    _diagnostics_harness,
    _fold_diagnostics,
    _roll_sensitivity_assertions,
    _select_top_cells,
    _smoke_null_diagnostics,
    _target_missing_reason_rows,
    _target_missing_reason_summary,
)
from k2_systematic_macro.signals.trade_candidates import (
    TradeCandidateConfig,
    build_trade_candidates,
)
from volbook.bundle import Bar
from volbook.datalake import MinuteLake
from volbook.continuous import FrontMonthCoverageError


def _bar(ts: datetime, c: float, v: float = 10.0) -> Bar:
    return Bar(t=ts.isoformat(), o=c - 0.05, h=c + 0.10, l=c - 0.10, c=c, v=v)


def _populate_cl_lake(db_path: Path, *, minutes: int = 12 * 60) -> None:
    base = datetime(2026, 5, 10, 22, 0, tzinfo=timezone.utc)  # 18:00 ET session start
    bars = [_bar(base + timedelta(minutes=i), 70.0 + i * 0.01) for i in range(minutes)]
    with MinuteLake(db_path) as lake:
        lake.upsert_bars("CL", bars, expiry="202606")


def _synthetic_feature_panel(rows: int = 260) -> pd.DataFrame:
    ts = pd.date_range("2026-01-01", periods=rows, freq="4h", tz="UTC")
    wave = np.sin(np.arange(rows) / 8.0)
    close = 70.0 + np.cumsum(0.05 * wave + 0.02 * np.sign(wave))
    bars = pd.DataFrame(
        {
            "ts": ts,
            "symbol": "CL",
            "expiry": "202606",
            "timeframe": "4h",
            "o": close - 0.05,
            "h": close + 0.30 + 0.05 * np.abs(wave),
            "l": close - 0.30 - 0.05 * np.abs(wave),
            "c": close,
            "v": 100.0 + np.arange(rows) % 20,
            "bar_count": 240,
            "session_date": ts.date.astype(str),
            "session_start_ts": ts,
            "session_tz": "America/New_York",
            "source_kind": "volbook",
        }
    )
    return build_feature_frame(
        bars,
        FeatureConfig(
            atr_lookback=5,
            realized_vol_lookbacks=(6, 12, 24),
            vol_of_vol_lookback=6,
            compression_lookback=12,
            return_horizons=(1, 3, 6, 12),
            breakout_lookback=10,
            trend_lookback=8,
        ),
    )


def test_volbook_adapter_loads_duckdb_and_resamples_session_bars(tmp_path: Path) -> None:
    db_path = tmp_path / "futures_market.duckdb"
    _populate_cl_lake(db_path, minutes=8 * 60)
    adapter = VolbookResearchDataAdapter(
        VolbookLoadSpec(source="duckdb", lake_path=db_path, symbol="CL", contract_source="dated_front")
    )

    minutes = adapter.load_minute_bars()
    bars_4h = adapter.resample(minutes, "4h")

    assert minutes.shape[0] == 8 * 60
    assert bars_4h.shape[0] == 2
    assert bars_4h["bar_count"].to_list() == [240, 240]
    assert set(bars_4h["symbol"]) == {"CL"}
    assert set(bars_4h["timeframe"]) == {"4h"}
    assert bars_4h["session_date"].iloc[0] == "2026-05-11"


def test_research_csv_export_strips_excel_prefixes_and_keeps_ohlc_numeric(tmp_path: Path) -> None:
    path = tmp_path / "bars_4h.csv"
    frame = pd.DataFrame(
        {
            "ts": ["'2026-05-11T02:00:00+00:00"],
            "symbol": ["CL"],
            "expiry": ["'202606"],
            "timeframe": ["4h"],
            "o": ["'70.1"],
            "h": ["'70.25"],
            "l": ["'69.95"],
            "c": ["'70.2"],
            "v": ["'1234"],
            "bar_count": ["'240"],
            "note": ["'keep literal text"],
        }
    )

    write_research_csv(frame, path)

    text = path.read_text()
    assert "'70.1" not in text
    assert "'2026-05-11" not in text
    assert "'202606" not in text
    assert "'keep literal text" in text

    exported = pd.read_csv(path)
    for col in ["o", "h", "l", "c", "v", "bar_count"]:
        assert pd.to_numeric(exported[col], errors="coerce").notna().all()
    assert exported.loc[0, "o"] == pytest.approx(70.1)
    assert exported.loc[0, "c"] == pytest.approx(70.2)


def test_volbook_adapter_auto_falls_back_to_duckdb(tmp_path: Path) -> None:
    db_path = tmp_path / "futures_market.duckdb"
    _populate_cl_lake(db_path, minutes=60)
    adapter = VolbookResearchDataAdapter(
        VolbookLoadSpec(
            source="auto",
            parquet_root=tmp_path / "missing_parquet",
            lake_path=db_path,
            symbol="CL",
            contract_source="dated_front",
        )
    )

    minutes = adapter.load_minute_bars()

    assert minutes.shape[0] == 60
    assert set(minutes["symbol"]) == {"CL"}


def test_volbook_adapter_loads_institutional_continuous_from_duckdb(tmp_path: Path) -> None:
    db_path = tmp_path / "futures_market.duckdb"
    with MinuteLake(db_path) as lake:
        may13 = datetime(2026, 5, 13, 14, 0, tzinfo=timezone.utc)
        may14 = datetime(2026, 5, 14, 14, 0, tzinfo=timezone.utc)
        may15 = datetime(2026, 5, 15, 14, 0, tzinfo=timezone.utc)
        lake.upsert_bars("CL", [_bar(may13, 69.0, 100.0), _bar(may14, 70.0, 100.0)], expiry="202606")
        lake.upsert_bars(
            "CL",
            [_bar(may13, 71.0, 200.0), _bar(may14, 72.0, 200.0), _bar(may15, 73.0, 250.0)],
            expiry="202607",
        )
    adapter = VolbookResearchDataAdapter(
        VolbookLoadSpec(
            source="duckdb",
            lake_path=db_path,
            symbol="CL",
            contract_source="institutional_continuous",
            continuous_volume_crossover_sessions=2,
        )
    )

    minutes = adapter.load_minute_bars()

    assert minutes["expiry"].to_list() == ["202606", "202607", "202607"]
    assert minutes["c"].to_list() == pytest.approx([71.0, 72.0, 73.0])
    assert adapter.last_load_metadata["construction"] == "institutional_continuous"
    assert adapter.last_load_metadata["front_month_coverage"]["status"] == "valid"
    assert adapter.last_roll_schedule.iloc[0]["trigger"] == "volume_crossover"


def test_volbook_adapter_institutional_continuous_rejects_far_dated_only_cl(tmp_path: Path) -> None:
    db_path = tmp_path / "futures_market.duckdb"
    with MinuteLake(db_path) as lake:
        ts = datetime(2025, 9, 15, 14, 0, tzinfo=timezone.utc)
        lake.upsert_bars("CL", [_bar(ts, 70.0)], expiry="202702")
    adapter = VolbookResearchDataAdapter(
        VolbookLoadSpec(
            source="duckdb",
            lake_path=db_path,
            symbol="CL",
            contract_source="institutional_continuous",
        )
    )

    with pytest.raises(FrontMonthCoverageError, match="eligible front range.*202510"):
        adapter.load_minute_bars()


def test_institutional_continuous_respects_research_start_bound(tmp_path: Path) -> None:
    db_path = tmp_path / "futures_market.duckdb"
    with MinuteLake(db_path) as lake:
        stale_ts = datetime(2022, 1, 13, 14, 0, tzinfo=timezone.utc)
        valid_ts = datetime(2025, 9, 15, 14, 0, tzinfo=timezone.utc)
        lake.upsert_bars("CL", [_bar(stale_ts, 70.0)], expiry="202712")
        lake.upsert_bars("CL", [_bar(valid_ts, 69.0, 100.0)], expiry="202510")
        lake.upsert_bars("CL", [_bar(valid_ts, 71.0, 90.0)], expiry="202511")
    adapter = VolbookResearchDataAdapter(
        VolbookLoadSpec(
            source="duckdb",
            lake_path=db_path,
            symbol="CL",
            contract_source="institutional_continuous",
            start_ts="2025-09-01T00:00:00+00:00",
        )
    )

    minutes = adapter.load_minute_bars()

    assert minutes["expiry"].to_list() == ["202510"]
    assert adapter.last_load_metadata["front_month_coverage"]["status"] == "valid"


def test_volbook_adapter_dated_front_rejects_far_dated_only_cl(tmp_path: Path) -> None:
    db_path = tmp_path / "futures_market.duckdb"
    with MinuteLake(db_path) as lake:
        ts = datetime(2025, 9, 15, 14, 0, tzinfo=timezone.utc)
        lake.upsert_bars("CL", [_bar(ts, 70.0)], expiry="202702")
    adapter = VolbookResearchDataAdapter(
        VolbookLoadSpec(source="duckdb", lake_path=db_path, symbol="CL", contract_source="dated_front")
    )

    with pytest.raises(FrontMonthCoverageError, match="eligible front range.*202510"):
        adapter.load_minute_bars()


def test_canonical_builder_returns_bars_features_and_lineage(tmp_path: Path) -> None:
    db_path = tmp_path / "futures_market.duckdb"
    _populate_cl_lake(db_path, minutes=30 * 60)
    config = CLResearchConfig(
        source="duckdb",
        lake_path=db_path,
        primary_timeframe="4h",
        secondary_timeframes=("1h", "1d"),
    )

    dataset = CanonicalCLDatasetBuilder(config).build()

    assert dataset.symbol == "CL"
    assert set(dataset.bars) == {"4h", "1h", "1d"}
    assert set(dataset.features) == {"4h", "1h", "1d"}
    assert dataset.metadata["source_system"] == "volbook"
    assert dataset.metadata["contract_source"] == "dated_front"
    assert dataset.metadata["primary_timeframe"] == "4h"


def test_feature_frame_uses_prior_breakout_channel() -> None:
    ts = pd.date_range("2026-01-01", periods=8, freq="4h", tz="UTC")
    bars = pd.DataFrame(
        {
            "ts": ts,
            "symbol": "CL",
            "expiry": "202606",
            "timeframe": "4h",
            "o": np.arange(8, dtype=float) + 70,
            "h": [71, 72, 73, 74, 75, 76, 77, 120],
            "l": [69, 70, 71, 72, 73, 74, 75, 76],
            "c": [70, 71, 72, 73, 74, 75, 76, 90],
            "v": 10.0,
        }
    )
    cfg = FeatureConfig(
        atr_lookback=2,
        realized_vol_lookbacks=(2,),
        vol_of_vol_lookback=2,
        compression_lookback=3,
        return_horizons=(1, 2),
        breakout_lookback=3,
        trend_lookback=3,
    )

    features = build_feature_frame(bars, cfg)

    last = features.iloc[-1]
    prior_high = max([75, 76, 77])
    assert last["breakout_distance_high_atr"] == pytest.approx(
        (90 - prior_high) / last["atr_2"]
    )
    assert last["breakout_distance_high_atr"] != pytest.approx((90 - 120) / last["atr_2"])


def test_quality_report_is_diagnostics_only(tmp_path: Path) -> None:
    db_path = tmp_path / "futures_market.duckdb"
    _populate_cl_lake(db_path, minutes=30 * 60)
    dataset = CanonicalCLDatasetBuilder(
        CLResearchConfig(source="duckdb", lake_path=db_path)
    ).build()

    report = build_cl_quality_report(dataset)

    assert "Coverage And Quality" in report
    assert "does not define entries, exits" in report
    assert "Dated-front output is unadjusted" in report
    assert "Median realized volatility" in report


def test_adapter_missing_lake_fails_clearly(tmp_path: Path) -> None:
    adapter = VolbookResearchDataAdapter(
        VolbookLoadSpec(source="duckdb", lake_path=tmp_path / "missing.duckdb", symbol="CL")
    )

    with pytest.raises(FileNotFoundError, match="volbook DuckDB lake not found"):
        adapter.load_minute_bars()


def test_target_generation_builds_forward_labels_without_feature_lookahead() -> None:
    features = _synthetic_feature_panel(120)
    targets = build_target_frame(
        features,
        TargetConfig(
            horizons=("4h", "1d"),
            atr_lookback=5,
            realized_vol_lookback=6,
            tail_atr_multiple=0.75,
            vol_expansion_threshold=1.0,
        ),
    )

    assert "target_future_move_norm_4h" in targets
    assert "target_future_vol_expansion_1d" in targets
    assert "target_tail_event_4h" in targets
    assert "target_conditional_direction_4h" in targets
    assert targets["target_future_move_norm_4h"].iloc[-1] != targets["target_future_move_norm_4h"].iloc[-1]
    assert set(targets["target_tail_event_4h"].dropna().unique()).issubset({0.0, 1.0})


def test_target_generation_covers_4h_vol_expansion_one_bar_horizon() -> None:
    features = _synthetic_feature_panel(120)
    targets = build_target_frame(
        features,
        TargetConfig(
            horizons=("4h",),
            atr_lookback=5,
            realized_vol_lookback=6,
            vol_expansion_threshold=1.0,
        ),
    )

    valid = targets["target_future_vol_expansion_4h"].dropna()

    assert not valid.empty
    assert valid.index.min() == 6
    assert valid.index.max() == 118
    assert valid.ge(0).all()
    assert set(targets["target_vol_expansion_event_4h"].dropna().unique()).issubset({0.0, 1.0})


def test_gmm_regimes_and_evaluation_summarize_regime_behavior() -> None:
    features = _synthetic_feature_panel(220)
    panel = build_target_frame(
        features,
        TargetConfig(atr_lookback=5, realized_vol_lookback=6, tail_atr_multiple=0.75),
    )
    regimes = fit_regimes(
        panel,
        RegimeConfig(
            n_states=3,
            min_samples=50,
            feature_columns=(
                "realized_vol_12",
                "realized_vol_24",
                "atr_compression",
                "range_compression",
                "trend_persistence",
                "trend_return_atr",
            ),
        ),
    )

    evaluation = evaluate_regimes(regimes)

    assert regimes["regime"].notna().sum() > 100
    assert set(regimes["regime"].dropna().unique()).issubset({0.0, 1.0, 2.0})
    assert not evaluation.transition_matrix.empty
    assert not evaluation.summary.empty
    assert "tail_participation" in evaluation.summary


def test_walk_forward_expansion_model_outputs_calibration() -> None:
    features = _synthetic_feature_panel(280)
    panel = build_target_frame(
        features,
        TargetConfig(
            atr_lookback=5,
            realized_vol_lookback=6,
            vol_expansion_threshold=0.95,
        ),
    )
    panel = fit_regimes(
        panel,
        RegimeConfig(
            n_states=2,
            min_samples=50,
            feature_columns=("realized_vol_12", "atr_compression", "trend_persistence"),
        ),
    )
    results = walk_forward_expansion_models(
        panel,
        ExpansionModelConfig(
            model_kinds=("logistic",),
            min_train_size=80,
            test_size=30,
            calibration_bins=3,
            feature_columns=(
                "realized_vol_12",
                "atr_compression",
                "trend_persistence",
                "regime",
                "regime_probability",
            ),
        ),
    )

    result = results["logistic"]
    assert result.metrics["n"] and result.metrics["n"] > 0
    assert not result.predictions.empty
    assert not result.calibration.empty
    assert {"mean_probability", "event_rate"}.issubset(result.calibration.columns)


def test_research_pipeline_writes_local_artifacts(tmp_path: Path) -> None:
    db_path = tmp_path / "futures_market.duckdb"
    _populate_cl_lake(db_path, minutes=90 * 60)
    result = run_cl_research_pipeline(
        CLResearchPipelineConfig(
            cl_config=CLResearchConfig(source="duckdb", lake_path=db_path),
            output_root=tmp_path / "artifacts",
            include_optional_boosters=False,
            expansion_config=ExpansionModelConfig(min_train_size=30, test_size=10),
        )
    )

    assert result.output_dir.exists()
    assert result.artifact_paths["report"].exists()
    assert result.artifact_paths["primary_panel"].exists()
    assert result.artifact_paths["model_metrics"].exists()


def test_target_missing_reason_classification_identifies_warmup_tail_and_roll() -> None:
    ts = pd.date_range("2026-01-01", periods=10, freq="4h", tz="UTC")
    panel = pd.DataFrame(
        {
            "ts": ts,
            "expiry": ["202602"] * 5 + ["202603"] * 5,
            "session_date": [str(x.date()) for x in ts],
            "target_future_vol_expansion_4h": [np.nan, 1.0, 1.1, 1.2, 1.3, np.nan, 1.4, 1.5, 1.6, np.nan],
        }
    )

    rows = _target_missing_reason_rows(panel, primary_timeframe="4h")
    summary = _target_missing_reason_summary(rows)

    assert set(rows["reason"]) == {"warmup", "roll_boundary", "tail_end_of_sample"}
    assert summary["missing_rows"].sum() == 3
    assert rows.loc[rows["reason"] == "roll_boundary", "is_roll_boundary"].all()


def test_roll_sensitivity_assertion_warns_when_sequence_changes_without_row_count_change() -> None:
    summary = pd.DataFrame(
        [
            {
                "active_expiry_sequence_changed": True,
                "panel_rows_changed": False,
            }
        ]
    )

    assertions = _roll_sensitivity_assertions(summary)

    row = assertions[assertions["check_id"] == "roll_sequence_changed_same_rows"].iloc[0]
    assert row["status"] == "WARN"


def test_fold_diagnostics_reports_spearman_vector_and_monotonic_share() -> None:
    predictions = pd.DataFrame(
        {
            "fold": [0] * 6 + [1] * 6,
            "y_true": [0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0],
            "y_prob": [0.1, 0.2, 0.3, 0.7, 0.8, 0.9, 0.9, 0.8, 0.7, 0.3, 0.2, 0.1],
        }
    )

    diagnostics = _fold_diagnostics(predictions, bins=3)

    assert "calibration_event_rate_vector" in diagnostics
    assert diagnostics.loc[diagnostics["fold"] == 0, "calibration_spearman"].iloc[0] > 0
    assert diagnostics.loc[diagnostics["fold"] == 0, "calibration_monotonic"].iloc[0]


def test_top_cell_selection_is_deterministic_with_min_n_and_tiebreakers() -> None:
    cells = pd.DataFrame(
        [
            {"regime": 1, "probability_decile": 9, "n": 20, "event_rate": 0.9, "mean_probability": 0.8, "convexity_research_score": 1.0},
            {"regime": 1, "probability_decile": 10, "n": 40, "event_rate": 0.6, "mean_probability": 0.7, "convexity_research_score": 1.0},
            {"regime": 0, "probability_decile": 10, "n": 40, "event_rate": 0.6, "mean_probability": 0.7, "convexity_research_score": 1.0},
        ]
    )

    selected = _select_top_cells(cells, TopCellSelectorConfig(min_n=30), max_cells=2)

    assert selected["n"].min() >= 30
    assert selected.iloc[0]["regime"] == 0
    assert selected.iloc[0]["probability_decile"] == 10


def test_convexity_score_is_reproducible_from_output_columns() -> None:
    score = _convexity_research_score(
        event_rate=0.4,
        mean_probability=0.6,
        tail_participation=0.2,
        drawdown_proxy=-0.1,
    )

    assert score == pytest.approx(1.1)


def test_smoke_null_diagnostics_and_harness_emit_statuses() -> None:
    predictions = pd.DataFrame(
        {
            "y_true": [0, 1, 0, 1, 0, 1, 0, 1],
            "y_prob": [0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4, 0.5],
        }
    )
    smoke = _smoke_null_diagnostics(predictions, seed=7)
    fold = _fold_diagnostics(predictions.assign(fold=0), bins=2)
    harness = _diagnostics_harness(
        target_missing_reasons=pd.DataFrame(),
        roll_assertions=pd.DataFrame(
            [{"check_id": "roll_sequence_changed_same_rows", "status": "PASS", "observed": False, "threshold": "warn"}]
        ),
        fold_diagnostics=fold,
        calibration_summary=pd.DataFrame(
            [{"share_monotonic": 1.0, "median_spearman": 1.0, "share_positive_spearman": 1.0}]
        ),
        selected_cells=pd.DataFrame([{"n": 50}]),
        smoke_null=smoke,
        thresholds=DiagnosticsThresholdConfig(smoke_auc_edge_max_abs=1.0, smoke_probability_label_corr_max_abs=1.0),
    )

    assert set(smoke["test"]) == {"shuffled_labels", "constant_base_rate_score"}
    assert {"PASS", "WARN", "FAIL"}.intersection(set(harness["status"]))


def _trade_candidate_fixture(
    *,
    rows: int = 80,
    high_regime: int = 1,
    high_move_sign: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ts = pd.date_range("2026-01-01", periods=rows, freq="4h", tz="UTC")
    y_prob = np.linspace(0.05, 0.98, rows)
    high = y_prob >= 0.82
    regime = np.where(high, high_regime, 0)
    panel = pd.DataFrame(
        {
            "ts": ts,
            "symbol": "CL",
            "c": 70.0 + np.arange(rows) * 0.05,
            "regime": regime,
            "regime_probability": 0.8,
            "target_vol_expansion_event_1d": high.astype(float),
            "target_future_return_1d": np.where(high, high_move_sign * 0.01, 0.0),
            "target_future_log_return_1d": np.where(high, high_move_sign * 0.01, 0.0),
            "target_future_move_norm_1d": np.where(high, high_move_sign * 1.2, 0.0),
            "target_tail_event_1d": high.astype(float),
            "target_conditional_direction_1d": np.where(high, high_move_sign, 0.0),
        }
    )
    predictions = pd.DataFrame(
        {
            "ts": ts,
            "fold": np.arange(rows) // 10,
            "y_true": high.astype(float),
            "y_prob": y_prob,
            "model_kind": "synthetic",
        }
    )
    harness = pd.DataFrame(
        [
            {"check_id": "smoke", "status": "PASS", "observed": True, "threshold": "pass"},
            {"check_id": "calibration", "status": "WARN", "observed": 0.4, "threshold": "warn"},
        ]
    )
    return panel, predictions, harness


def test_trade_candidates_emit_long_candidate_from_prior_directional_cell() -> None:
    panel, predictions, harness = _trade_candidate_fixture(high_move_sign=1.0)

    candidates, directional = build_trade_candidates(
        primary_panel=panel,
        predictions=predictions,
        diagnostics_harness=harness,
        symbol="CL",
        config=TradeCandidateConfig(
            min_cell_n=3,
            expansion_probability_min=0.80,
            expansion_decile_min=9,
            min_directional_confidence=0.55,
            min_directional_asymmetry=0.10,
        ),
    )

    latest = candidates.iloc[-1]
    assert latest["action"] == "long_candidate"
    assert latest["reason_codes"] == "pass"
    assert latest["directional_prob_up"] == pytest.approx(1.0)
    assert not directional.empty


def test_trade_candidates_emit_short_candidate_from_prior_directional_cell() -> None:
    panel, predictions, harness = _trade_candidate_fixture(high_move_sign=-1.0)

    candidates, _ = build_trade_candidates(
        primary_panel=panel,
        predictions=predictions,
        diagnostics_harness=harness,
        symbol="CL",
        config=TradeCandidateConfig(
            min_cell_n=3,
            expansion_probability_min=0.80,
            expansion_decile_min=9,
            min_directional_confidence=0.55,
            min_directional_asymmetry=0.10,
        ),
    )

    latest = candidates.iloc[-1]
    assert latest["action"] == "short_candidate"
    assert latest["directional_prob_down"] == pytest.approx(1.0)


def test_trade_candidates_suppress_when_cell_sample_or_diagnostics_fail() -> None:
    panel, predictions, harness = _trade_candidate_fixture()
    failing_harness = harness.assign(status=["FAIL", "WARN"])

    candidates, _ = build_trade_candidates(
        primary_panel=panel,
        predictions=predictions,
        diagnostics_harness=failing_harness,
        symbol="CL",
        config=TradeCandidateConfig(min_cell_n=3, expansion_probability_min=0.80, expansion_decile_min=9),
    )
    sparse, _ = build_trade_candidates(
        primary_panel=panel,
        predictions=predictions,
        diagnostics_harness=harness,
        symbol="CL",
        config=TradeCandidateConfig(min_cell_n=100, expansion_probability_min=0.80, expansion_decile_min=9),
    )

    assert set(candidates["action"]) == {"no_trade"}
    assert candidates["reason_codes"].str.contains("diagnostics_fail").any()
    assert set(sparse["action"]) == {"no_trade"}
    assert sparse["reason_codes"].str.contains("cell_sample_below_min").any()


def test_trade_candidates_regime2_overlay_defaults_to_risk_off() -> None:
    panel, predictions, harness = _trade_candidate_fixture(high_regime=2)

    risk_off, _ = build_trade_candidates(
        primary_panel=panel,
        predictions=predictions,
        diagnostics_harness=harness,
        symbol="CL",
        config=TradeCandidateConfig(
            min_cell_n=3,
            expansion_probability_min=0.80,
            expansion_decile_min=9,
            regime2_policy="risk_off_no_entry",
        ),
    )
    size_down, _ = build_trade_candidates(
        primary_panel=panel,
        predictions=predictions,
        diagnostics_harness=harness,
        symbol="CL",
        config=TradeCandidateConfig(
            min_cell_n=3,
            expansion_probability_min=0.80,
            expansion_decile_min=9,
            regime2_policy="size_down_50pct",
        ),
    )

    assert risk_off.iloc[-1]["action"] == "no_trade"
    assert "regime2_risk_off_no_entry" in risk_off.iloc[-1]["reason_codes"]
    assert size_down.iloc[-1]["action"] == "long_candidate"
    assert size_down.iloc[-1]["confidence_overlay_multiplier"] == pytest.approx(0.5)


def test_trade_candidates_are_deterministic() -> None:
    panel, predictions, harness = _trade_candidate_fixture()
    config = TradeCandidateConfig(min_cell_n=3, expansion_probability_min=0.80, expansion_decile_min=9)

    first, first_directional = build_trade_candidates(
        primary_panel=panel,
        predictions=predictions,
        diagnostics_harness=harness,
        symbol="CL",
        config=config,
    )
    second, second_directional = build_trade_candidates(
        primary_panel=panel,
        predictions=predictions,
        diagnostics_harness=harness,
        symbol="CL",
        config=config,
    )

    pd.testing.assert_frame_equal(first, second)
    pd.testing.assert_frame_equal(first_directional, second_directional)
