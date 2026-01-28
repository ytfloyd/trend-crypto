#!/usr/bin/env python
from __future__ import annotations

import argparse
import ast
from pathlib import Path
from typing import Iterable

import numpy as np

from alphas.primitives import PRIMITIVES


DEFAULT_COLUMNS = ["close", "open", "high", "low", "volume"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate formulaic alpha candidates (v0-safe)")
    p.add_argument("--output", default="alphas_batch_001.txt", help="Output file")
    p.add_argument("--count", type=int, default=500, help="Number of formulas to generate")
    p.add_argument("--seed", type=int, default=0, help="RNG seed")
    p.add_argument("--windows", default="5,10,20,60", help="Comma-separated windows")
    p.add_argument("--columns", default=None, help="Comma-separated columns")
    p.add_argument("--allow-vwap", action="store_true", help="Allow vwap column")
    p.add_argument("--force", action="store_true", help="Allow count > 500")
    return p.parse_args()


def _parse_windows(value: str) -> list[int]:
    out = []
    for tok in value.split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(int(tok))
    return out


def _parse_columns(value: str | None, allow_vwap: bool) -> list[str]:
    cols = DEFAULT_COLUMNS if not value else [c.strip() for c in value.split(",") if c.strip()]
    if allow_vwap and "vwap" not in cols:
        cols.append("vwap")
    return cols


def _extract_names(formula: str) -> tuple[set[str], set[str]]:
    tree = ast.parse(formula, mode="eval")
    ops = set()
    cols = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            ops.add(node.func.id)
        if isinstance(node, ast.Name):
            cols.add(node.id)
    return ops, cols


def _validate_formula(formula: str, valid_ops: set[str], valid_cols: set[str]) -> bool:
    ops, cols = _extract_names(formula)
    if "ts_rank" in ops:
        return False
    if not ops.issubset(valid_ops):
        return False
    # remove op names from cols
    cols = cols - ops
    return cols.issubset(valid_cols)


def _cluster_liquidity(w: int) -> list[str]:
    return [
        f"correlation(rank(delta(close, {w})), rank(volume), {w})",
        f"-1 * correlation(rank(delta(close, {w})), rank(volume), {w})",
        f"correlation(rank(delta(log(close), {w})), rank(volume), {w})",
        f"-1 * correlation(rank(delta(log(close), {w})), rank(volume), {w})",
    ]


def _cluster_reversion(w: int) -> list[str]:
    return [
        f"-1 * delta(close, {w})",
        f"-1 * delta(log(close), {w})",
        f"-1 * delta(open, {w})",
    ]


def _cluster_volatility(w: int) -> list[str]:
    return [
        f"correlation(rank(abs(delta(close, {w}))), rank(high - low), {w})",
        f"correlation(rank(abs(delta(log(close), {w}))), rank(high - low), {w})",
    ]


def _build_candidates(windows: Iterable[int]) -> dict[str, list[str]]:
    clusters = {
        "Liquidity": [],
        "Reversion": [],
        "Volatility": [],
    }
    for w in windows:
        clusters["Liquidity"].extend(_cluster_liquidity(w))
        clusters["Reversion"].extend(_cluster_reversion(w))
        clusters["Volatility"].extend(_cluster_volatility(w))
    return clusters


def generate_formulas(
    *,
    count: int,
    seed: int,
    windows: list[int],
    columns: list[str],
) -> tuple[list[tuple[str, str, str]], set[str]]:
    valid_ops = set(PRIMITIVES.keys())
    valid_cols = set(columns)

    clusters = _build_candidates(windows)
    valid = []
    ops_used: set[str] = set()

    for cluster_name, formulas in clusters.items():
        for formula in formulas:
            if _validate_formula(formula, valid_ops, valid_cols):
                valid.append((cluster_name, formula))
                ops, _ = _extract_names(formula)
                ops_used |= ops

    if not valid:
        raise ValueError("No valid formulas generated. Check primitives/columns.")

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(valid)).tolist()
    pool = [valid[i] for i in order]

    results: list[tuple[str, str, str]] = []
    while len(results) < count:
        for cluster_name, formula in pool:
            results.append((cluster_name, formula, formula))
            if len(results) >= count:
                break
        if len(results) >= count:
            break
        # shuffle again for more diversity if count exceeds pool
        order = rng.permutation(len(valid)).tolist()
        pool = [valid[i] for i in order]

    return results[:count], ops_used


def write_output(path: Path, rows: list[tuple[str, str, str]]) -> None:
    clusters = {"Liquidity": [], "Reversion": [], "Volatility": []}
    for cluster, _, formula in rows:
        clusters[cluster].append(formula)

    with open(path, "w", encoding="utf-8") as f:
        alpha_idx = 1
        for cluster in ["Liquidity", "Reversion", "Volatility"]:
            f.write(f"# {cluster}\n")
            for formula in clusters[cluster]:
                f.write(f"alpha_gen_{alpha_idx:04d} = {formula}\n")
                alpha_idx += 1


def main() -> None:
    args = parse_args()
    if args.count > 500 and not args.force:
        raise SystemExit("Count exceeds 500. Use --force to override.")

    windows = _parse_windows(args.windows)
    columns = _parse_columns(args.columns, args.allow_vwap)

    rows, ops_used = generate_formulas(
        count=args.count,
        seed=args.seed,
        windows=windows,
        columns=columns,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_output(out_path, rows)

    print("[generate_formulas] Done")
    print(f"  output: {out_path}")
    print(f"  generated: {len(rows)}")
    print(f"  ops_used: {', '.join(sorted(ops_used))}")
    print(f"  windows: {windows}")


if __name__ == "__main__":
    main()
