import duckdb
import pytest

from scripts.run_alpha_factory import resolve_price_table


def _make_db(path):
    con = duckdb.connect(str(path))
    con.execute("CREATE TABLE bars_1d_clean(ts TIMESTAMP, symbol VARCHAR, close DOUBLE, volume DOUBLE)")
    con.close()


def test_resolve_price_table_missing_without_fallback(tmp_path):
    db_path = tmp_path / "test.duckdb"
    _make_db(db_path)

    with pytest.raises(SystemExit) as exc:
        resolve_price_table(str(db_path), "bars_1d_usd_universe_clean_adv10m")

    msg = str(exc.value)
    assert "Requested table" in msg
    assert "create_usd_universe_adv10m_view.py" in msg


def test_resolve_price_table_missing_with_fallback(tmp_path):
    db_path = tmp_path / "test.duckdb"
    _make_db(db_path)

    table = resolve_price_table(
        str(db_path),
        "bars_1d_usd_universe_clean_adv10m",
        fallback="bars_1d_clean",
        allow_fallback=True,
    )
    assert table == "bars_1d_clean"
