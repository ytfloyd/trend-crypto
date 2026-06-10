"""Tests for the CME product-universe builder."""
from __future__ import annotations

import json

from scripts.reference.build_cme_universe_from_docx import (
    DEFAULT_SOURCE,
    parse_source,
    render_markdown,
    write_outputs,
)


def _by_name():
    meta, products = parse_source(DEFAULT_SOURCE)
    return meta, {p.product_name: p for p in products}, products


def test_parse_docx_extracts_source_metadata_and_rows() -> None:
    meta, by_name, products = _by_name()

    assert meta["source_format"] == "xlsx"
    assert meta["sheet_name"] == "Product Slate Apr 25 2026"
    assert meta["trade_date"] == "Apr 25 2026"
    assert meta["row_count"] == 3032
    assert len(products) == 3032
    assert "Three-Month SOFR Futures" in by_name
    assert "E-mini Russell 2000 Index Futures" in by_name


def test_known_high_liquidity_products_parse_exact_values() -> None:
    _, by_name, _ = _by_name()

    sofr = by_name["Three-Month SOFR Futures"]
    es = by_name["E-mini S&P 500 Futures"]
    cl = by_name["Crude Oil Futures"]

    assert sofr.volume == 3_921_027
    assert sofr.open_interest == 12_006_141
    assert sofr.root_symbol == "SR3"
    assert sofr.cleared_as == "Futures"

    assert es.volume == 1_865_765
    assert es.open_interest == 1_964_936
    assert es.root_symbol == "ES"

    assert cl.volume == 1_083_155
    assert cl.open_interest == 1_990_645
    assert cl.exchange == "NYMEX"
    assert cl.account_20m_tradable is True
    assert cl.account_20m_capacity_contracts >= 10


def test_products_are_ranked_by_tradability_score() -> None:
    _, _, products = _by_name()
    scores = [p.tradability_score for p in products[:100]]

    assert scores == sorted(scores, reverse=True)
    assert products[0].product_name == "Three-Month SOFR Futures"
    assert products[0].tier == "S"
    assert products[0].account_20m_tradable is True


def test_render_markdown_contains_required_sections() -> None:
    meta, _, products = _by_name()
    md = render_markdown(meta, products)

    assert "# CME Tradable Products Universe" in md
    assert "Top 50 Products By Tradability" in md
    assert "Top Futures Products" in md
    assert "Top Options Products" in md
    assert "$20M Account Tradability Screen" in md
    assert "$20M Account Tradable Products" in md
    assert "Asset Class Summary" in md
    assert "Product Slate Apr 25 2026" in md
    assert "Three-Month SOFR Futures" in md
    assert "E-mini S&P 500 Futures" in md


def test_write_outputs_creates_json_and_markdown(tmp_path) -> None:
    meta, _, products = _by_name()
    json_path = tmp_path / "cme.json"
    md_path = tmp_path / "cme.md"

    out_json, out_md = write_outputs(products[:10], meta, json_path=json_path, markdown_path=md_path)

    assert out_json == json_path
    assert out_md == md_path
    payload = json.loads(json_path.read_text())
    assert payload["meta"]["trade_date"] == "Apr 25 2026"
    assert payload["meta"]["source_format"] == "xlsx"
    assert payload["meta"]["account_20m_screen"]["account_size_usd"] == 20_000_000
    assert len(payload["products"]) == 10
    assert "account_20m_tradable" in payload["products"][0]
    assert md_path.read_text().startswith("# CME Tradable Products Universe")


def test_account_20m_screen_includes_all_b_plus_liquid_products() -> None:
    _, by_name, products = _by_name()
    included = [p for p in products if p.account_20m_tradable]

    assert len(included) == 158
    assert sum(p.cleared_as == "Futures" for p in included) == 92
    assert sum(p.cleared_as == "Options" for p in included) == 66
    assert by_name["E-mini S&P 500 Futures"].account_20m_tradable is True
    assert by_name["Crude Oil Futures"].account_20m_tradable is True
    assert by_name["Lumber Futures"].account_20m_tradable is True
