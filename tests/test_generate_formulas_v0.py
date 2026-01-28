import ast

from alphas.primitives import PRIMITIVES
from scripts.research.generate_formulas import generate_formulas


def _extract_ops_cols(formula: str) -> tuple[set[str], set[str]]:
    tree = ast.parse(formula, mode="eval")
    ops = set()
    cols = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            ops.add(node.func.id)
        if isinstance(node, ast.Name):
            cols.add(node.id)
    cols = cols - ops
    return ops, cols


def test_generate_formulas_v0_safe():
    rows, _ = generate_formulas(
        count=50,
        seed=42,
        windows=[5, 10],
        columns=["close", "open", "high", "low", "volume"],
    )

    valid_ops = set(PRIMITIVES.keys())
    valid_cols = {"close", "open", "high", "low", "volume"}

    for _, _, formula in rows:
        assert "ts_rank" not in formula
        ops, cols = _extract_ops_cols(formula)
        assert ops.issubset(valid_ops)
        assert cols.issubset(valid_cols)
