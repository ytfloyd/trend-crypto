from scripts.research import strategy_registry_v0 as reg


def test_validate_canonical_missing_tearsheet_pdf():
    strategy = {
        "id": "canonical_missing_tearsheet",
        "family": "test",
        "version": "v0",
        "status": "canonical",
        "equity_csv": "artifacts/equity.csv",
        "metrics_csv": "artifacts/metrics.csv",
        "tearsheet_pdf": "",
        "run_recipe": ["echo run"],
        "canonical_period": {"start": "2020-01-01", "end": "2020-12-31"},
    }
    errs = reg._validate_status_fields(strategy)
    assert any("tearsheet_pdf" in err for err in errs)


def test_validate_canonical_missing_tearsheet_step():
    strategy = {
        "id": "canonical_missing_tearsheet_step",
        "family": "test",
        "version": "v0",
        "status": "canonical",
        "equity_csv": "artifacts/equity.csv",
        "metrics_csv": "artifacts/metrics.csv",
        "tearsheet_pdf": "artifacts/tearsheet.pdf",
        "run_recipe": ["echo run"],
        "canonical_period": {"start": "2020-01-01", "end": "2020-12-31"},
    }
    errs = reg._validate_status_fields(strategy)
    assert any("tearsheet generation" in err for err in errs)
