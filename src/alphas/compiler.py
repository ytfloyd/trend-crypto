from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from alphas.parser import ExprNode, parse
from alphas.primitives import DOMAIN_CS, DOMAIN_TS, PRIMITIVES


@dataclass(frozen=True)
class ExecutionPlan:
    stage1_exprs: list[pl.Expr]
    stage2_exprs: list[pl.Expr]
    warmup_bars: dict[str, int]
    formulas: dict[str, str]


def compile_formulas(formulas: list[tuple[str, str]]) -> ExecutionPlan:
    parsed: list[tuple[str, str, ExprNode]] = []
    for name, formula in formulas:
        node = parse(formula)
        parsed.append((name, formula, node))

    ts_cache: dict[str, str] = {}
    stage1_exprs: list[pl.Expr] = []
    stage2_exprs: list[pl.Expr] = []
    warmup_bars: dict[str, int] = {}
    formulas_map: dict[str, str] = {name: formula for name, formula, _ in parsed}

    def _ensure_ts_expr(node: ExprNode) -> str:
        if node.signature in ts_cache:
            return ts_cache[node.signature]
        col_name = f"_ts_{len(ts_cache)}"
        ts_cache[node.signature] = col_name
        stage1_exprs.append(node.expr.alias(col_name))
        return col_name

    for name, _, node in parsed:
        warmup_bars[name] = node.warmup
        if node.domain == DOMAIN_TS:
            ts_col = _ensure_ts_expr(node)
            stage2_exprs.append(pl.col(ts_col).alias(name))
            continue

        ts_col_map: dict[str, str] = {}

        def _stage2_expr(curr: ExprNode) -> pl.Expr:
            if curr.domain == DOMAIN_TS:
                col = ts_col_map.get(curr.signature)
                if col is None:
                    col = _ensure_ts_expr(curr)
                    ts_col_map[curr.signature] = col
                return pl.col(col)

            if curr.kind == "binop":
                left = _stage2_expr(curr.args[0])
                right = _stage2_expr(curr.args[1])
                return _apply_binop(curr.op, left, right)

            if curr.kind == "unary":
                child = _stage2_expr(curr.args[0])
                return -child if curr.op == "-" else child

            if curr.kind == "call":
                fn_name = curr.name
                if fn_name is None:
                    raise ValueError("Malformed call node.")
                spec = PRIMITIVES[fn_name]
                built_args = [_stage2_expr(a) for a in curr.args]
                if fn_name in {"delta", "delay", "ts_min", "ts_max"}:
                    window = _require_window_arg(curr.args[1])
                    return spec.func(built_args[0], window)
                if fn_name in {"correlation", "covariance"}:
                    window = _require_window_arg(curr.args[2])
                    return spec.func(built_args[0], built_args[1], window)
                return spec.func(built_args[0])

            raise ValueError("Unsupported node in stage2 compilation.")

        stage2_exprs.append(_stage2_expr(node).alias(name))

    return ExecutionPlan(stage1_exprs, stage2_exprs, warmup_bars, formulas_map)


def _apply_binop(op: str | None, left: pl.Expr, right: pl.Expr) -> pl.Expr:
    if op == "+":
        return left + right
    if op == "-":
        return left - right
    if op == "*":
        return left * right
    if op == "/":
        return left / right
    raise ValueError("Unsupported binary operator.")


def _require_window_arg(node: ExprNode) -> int:
    if node.kind != "const" or node.value is None:
        raise ValueError("Window argument must be a numeric constant.")
    window = int(node.value)
    if window <= 0:
        raise ValueError("Window argument must be > 0.")
    return window
