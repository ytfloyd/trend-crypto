from __future__ import annotations

import ast
import re
from dataclasses import dataclass

import polars as pl

from alphas.primitives import PRIMITIVES, DOMAIN_CS, DOMAIN_TS


_ALPHA_NAME_RE = re.compile(r"^alpha_\d{3}$")


@dataclass(frozen=True)
class ExprNode:
    kind: str
    expr: pl.Expr
    domain: str
    warmup: int
    signature: str
    name: str | None = None
    value: float | int | None = None
    op: str | None = None
    args: tuple["ExprNode", ...] = ()


def load_alphas_file(path: str) -> list[tuple[str, str]]:
    formulas: list[tuple[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if "#" in stripped:
                stripped = stripped.split("#", 1)[0].strip()
            if not stripped:
                continue
            if "=" not in stripped:
                raise ValueError(f"Invalid alpha line (missing '='): {line.strip()}")
            name, formula = [s.strip() for s in stripped.split("=", 1)]
            if not _ALPHA_NAME_RE.match(name):
                raise ValueError(
                    f"Invalid alpha name '{name}'. Expected format 'alpha_###'."
                )
            if not formula:
                raise ValueError(f"Empty formula for {name}.")
            formulas.append((name, formula))
    if not formulas:
        raise ValueError(f"No alpha formulas found in {path}.")
    return formulas


def parse(formula: str) -> ExprNode:
    try:
        tree = ast.parse(formula, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Invalid formula syntax: {formula}") from exc

    return _parse_node(tree.body)


def _parse_node(node: ast.AST) -> ExprNode:
    if isinstance(node, ast.BinOp):
        left = _parse_node(node.left)
        right = _parse_node(node.right)
        op = _binop_symbol(node.op)
        expr = _apply_binop(op, left.expr, right.expr)
        domain = _merge_domain(left.domain, right.domain)
        warmup = max(left.warmup, right.warmup)
        signature = f"({op},{left.signature},{right.signature})"
        return ExprNode("binop", expr, domain, warmup, signature, op=op, args=(left, right))

    if isinstance(node, ast.UnaryOp):
        if not isinstance(node.op, (ast.USub, ast.UAdd)):
            raise ValueError("Unsupported unary operator.")
        child = _parse_node(node.operand)
        op = "-" if isinstance(node.op, ast.USub) else "+"
        expr = -child.expr if op == "-" else child.expr
        signature = f"({op},{child.signature})"
        return ExprNode("unary", expr, child.domain, child.warmup, signature, op=op, args=(child,))

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only simple function calls are allowed.")
        fn_name = node.func.id
        if fn_name not in PRIMITIVES:
            raise ValueError(f"Unknown primitive: {fn_name}")
        if node.keywords:
            raise ValueError("Keyword arguments are not allowed in formulas.")
        args = [_parse_node(a) for a in node.args]
        spec = PRIMITIVES[fn_name]

        _validate_call_arity(fn_name, args)

        if spec.domain == DOMAIN_TS and any(a.domain == DOMAIN_CS for a in args):
            raise ValueError(f"TS primitive '{fn_name}' cannot take CS input.")

        expr = _build_call_expr(fn_name, args)
        warmup = _compute_warmup(fn_name, args)
        domain = DOMAIN_CS if (spec.domain == DOMAIN_CS or any(a.domain == DOMAIN_CS for a in args)) else DOMAIN_TS
        signature = f"call:{fn_name}({','.join(a.signature for a in args)})"
        return ExprNode("call", expr, domain, warmup, signature, name=fn_name, args=tuple(args))

    if isinstance(node, ast.Name):
        name = node.id
        expr = pl.col(name)
        return ExprNode("col", expr, DOMAIN_TS, 0, f"col:{name}", name=name)

    if isinstance(node, ast.Constant):
        if not isinstance(node.value, (int, float)):
            raise ValueError("Only numeric constants are allowed.")
        expr = pl.lit(node.value)
        return ExprNode("const", expr, DOMAIN_TS, 0, f"const:{node.value}", value=node.value)

    raise ValueError(f"Unsupported syntax: {type(node).__name__}")


def _binop_symbol(op: ast.operator) -> str:
    if isinstance(op, ast.Add):
        return "+"
    if isinstance(op, ast.Sub):
        return "-"
    if isinstance(op, ast.Mult):
        return "*"
    if isinstance(op, ast.Div):
        return "/"
    raise ValueError("Unsupported binary operator.")


def _apply_binop(symbol: str, left: pl.Expr, right: pl.Expr) -> pl.Expr:
    if symbol == "+":
        return left + right
    if symbol == "-":
        return left - right
    if symbol == "*":
        return left * right
    if symbol == "/":
        return left / right
    raise ValueError("Unsupported binary operator.")


def _merge_domain(a: str, b: str) -> str:
    return DOMAIN_CS if (a == DOMAIN_CS or b == DOMAIN_CS) else DOMAIN_TS


def _validate_call_arity(fn_name: str, args: list[ExprNode]) -> None:
    if fn_name in {"abs", "log", "sign", "rank", "scale"}:
        if len(args) != 1:
            raise ValueError(f"{fn_name} expects 1 argument.")
    elif fn_name in {"delta", "delay", "ts_min", "ts_max"}:
        if len(args) != 2:
            raise ValueError(f"{fn_name} expects 2 arguments.")
    elif fn_name in {"correlation", "covariance"}:
        if len(args) != 3:
            raise ValueError(f"{fn_name} expects 3 arguments.")
    else:
        raise ValueError(f"Unsupported primitive: {fn_name}")


def _build_call_expr(fn_name: str, args: list[ExprNode]) -> pl.Expr:
    spec = PRIMITIVES[fn_name]
    if fn_name in {"delta", "delay", "ts_min", "ts_max"}:
        window = _require_int_constant(args[1])
        return spec.func(args[0].expr, window)
    if fn_name in {"correlation", "covariance"}:
        window = _require_int_constant(args[2])
        return spec.func(args[0].expr, args[1].expr, window)
    return spec.func(args[0].expr)


def _require_int_constant(node: ExprNode) -> int:
    if node.kind != "const" or not isinstance(node.value, (int, float)):
        raise ValueError("Window argument must be a numeric constant.")
    window = int(node.value)
    if window <= 0:
        raise ValueError("Window argument must be > 0.")
    return window


def _compute_warmup(fn_name: str, args: list[ExprNode]) -> int:
    if fn_name in {"delta", "delay", "ts_min", "ts_max"}:
        return max(args[0].warmup, _require_int_constant(args[1]))
    if fn_name in {"correlation", "covariance"}:
        return max(args[0].warmup, args[1].warmup, _require_int_constant(args[2]))
    return max(a.warmup for a in args) if args else 0
