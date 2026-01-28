import pytest
import polars as pl

from alphas.parser import parse


def test_parser_rejects_attributes():
    with pytest.raises(ValueError):
        parse("close.mean()")


def test_parser_rejects_subscript():
    with pytest.raises(ValueError):
        parse("close[0]")


def test_parser_rejects_unknown_primitive():
    with pytest.raises(ValueError):
        parse("foo(close)")


def test_parser_name_resolves_to_col():
    node = parse("close")
    assert isinstance(node.expr, pl.Expr)
    assert node.kind == "col"
    assert node.name == "close"
