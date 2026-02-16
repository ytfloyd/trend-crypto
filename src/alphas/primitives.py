from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import polars as pl


DOMAIN_TS = "TS"
DOMAIN_CS = "CS"


@dataclass(frozen=True)
class PrimitiveSpec:
    name: str
    domain: str
    func: Callable


# -------------------------
# Time-series primitives
# -------------------------

def ts_abs(x: pl.Expr) -> pl.Expr:
    return x.abs()


def ts_log(x: pl.Expr) -> pl.Expr:
    return x.log()


def ts_sign(x: pl.Expr) -> pl.Expr:
    return x.sign()


def ts_delay(x: pl.Expr, d: int) -> pl.Expr:
    return x.shift(d).over("symbol")


def ts_delta(x: pl.Expr, d: int) -> pl.Expr:
    return x - x.shift(d).over("symbol")


def ts_min(x: pl.Expr, d: int) -> pl.Expr:
    return x.rolling_min(window_size=d, min_samples=d).over("symbol")


def ts_max(x: pl.Expr, d: int) -> pl.Expr:
    return x.rolling_max(window_size=d, min_samples=d).over("symbol")


def ts_covariance(x: pl.Expr, y: pl.Expr, d: int) -> pl.Expr:
    mean_x = x.rolling_mean(window_size=d, min_samples=d).over("symbol")
    mean_y = y.rolling_mean(window_size=d, min_samples=d).over("symbol")
    mean_xy = (x * y).rolling_mean(window_size=d, min_samples=d).over("symbol")
    cov = mean_xy - mean_x * mean_y
    return cov


def ts_correlation(x: pl.Expr, y: pl.Expr, d: int) -> pl.Expr:
    mean_x = x.rolling_mean(window_size=d, min_samples=d).over("symbol")
    mean_y = y.rolling_mean(window_size=d, min_samples=d).over("symbol")
    mean_xy = (x * y).rolling_mean(window_size=d, min_samples=d).over("symbol")
    var_x = x.rolling_var(window_size=d, min_samples=d).over("symbol")
    var_y = y.rolling_var(window_size=d, min_samples=d).over("symbol")
    cov = mean_xy - mean_x * mean_y
    denom = (var_x.sqrt() * var_y.sqrt())
    return (
        pl.when((var_x > 0) & (var_y > 0))
        .then(cov / denom)
        .otherwise(None)
    )


# -------------------------
# Cross-sectional primitives
# -------------------------

def cs_rank(x: pl.Expr) -> pl.Expr:
    count = pl.len().over("ts")
    rank = x.rank(method="average").over("ts")
    return (
        pl.when(count > 1)
        .then((rank - 1) / (count - 1))
        .otherwise(0.0)
    )


def cs_scale(x: pl.Expr) -> pl.Expr:
    denom = x.abs().sum().over("ts")
    return x / (denom + 1e-12)


PRIMITIVES: dict[str, PrimitiveSpec] = {
    "abs": PrimitiveSpec("abs", DOMAIN_TS, ts_abs),
    "log": PrimitiveSpec("log", DOMAIN_TS, ts_log),
    "sign": PrimitiveSpec("sign", DOMAIN_TS, ts_sign),
    "delta": PrimitiveSpec("delta", DOMAIN_TS, ts_delta),
    "delay": PrimitiveSpec("delay", DOMAIN_TS, ts_delay),
    "ts_min": PrimitiveSpec("ts_min", DOMAIN_TS, ts_min),
    "ts_max": PrimitiveSpec("ts_max", DOMAIN_TS, ts_max),
    "correlation": PrimitiveSpec("correlation", DOMAIN_TS, ts_correlation),
    "covariance": PrimitiveSpec("covariance", DOMAIN_TS, ts_covariance),
    "rank": PrimitiveSpec("rank", DOMAIN_CS, cs_rank),
    "scale": PrimitiveSpec("scale", DOMAIN_CS, cs_scale),
}


TS_PRIMITIVES = {k for k, v in PRIMITIVES.items() if v.domain == DOMAIN_TS}
CS_PRIMITIVES = {k for k, v in PRIMITIVES.items() if v.domain == DOMAIN_CS}
