"""Signal processing pipeline for alpha post-processing.

Applies cross-sectional normalization, EMA smoothing, and outlier
winsorization to raw alpha signals.  Designed as an opt-in pipeline
stage between alpha generation and portfolio construction.

All transforms are disabled by default (pass-through mode).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import polars as pl


@dataclass(frozen=True)
class SignalProcessorConfig:
    """Configuration for the signal processing pipeline.

    Attributes
    ----------
    normalize : bool
        Apply cross-sectional z-score normalization per timestep.
    ema_halflife : float | None
        EMA half-life in bars.  None disables smoothing.
    winsor_threshold : float | None
        Clip signal values beyond +/- this many standard deviations.
        None disables winsorization.
    """

    normalize: bool = False
    ema_halflife: Optional[float] = None
    winsor_threshold: Optional[float] = None

    @property
    def is_passthrough(self) -> bool:
        return (
            not self.normalize
            and self.ema_halflife is None
            and self.winsor_threshold is None
        )


class SignalProcessor:
    """Applies sequential transforms to a wide-format signal DataFrame.

    Parameters
    ----------
    config : SignalProcessorConfig
        Pipeline configuration.  All transforms disabled by default.

    The expected input is a Polars DataFrame with columns:
        ts (Date/Datetime), symbol (Utf8), signal (Float64)
    or a wide-format DataFrame with ts as the first column and one column
    per symbol containing signal values.
    """

    def __init__(self, config: Optional[SignalProcessorConfig] = None) -> None:
        self.config = config or SignalProcessorConfig()

    def process_wide(self, df: pl.DataFrame) -> pl.DataFrame:
        """Process a wide-format signal DataFrame.

        Parameters
        ----------
        df : pl.DataFrame
            Wide format: first column is ``ts``, remaining columns are
            symbol signal values.

        Returns
        -------
        pl.DataFrame
            Processed signals in the same wide format.
        """
        if self.config.is_passthrough:
            return df

        ts_col = df.columns[0]
        signal_cols = [c for c in df.columns if c != ts_col]

        if not signal_cols:
            return df

        result = df.clone()

        if self.config.normalize:
            result = _normalize_cross_sectional(result, ts_col, signal_cols)

        if self.config.winsor_threshold is not None:
            result = _winsorize(result, signal_cols, self.config.winsor_threshold)

        if self.config.ema_halflife is not None:
            result = _ema_smooth(result, signal_cols, self.config.ema_halflife)

        return result

    def process_long(self, df: pl.DataFrame) -> pl.DataFrame:
        """Process a long-format signal DataFrame.

        Parameters
        ----------
        df : pl.DataFrame
            Long format with columns: ``ts``, ``symbol``, ``signal``.

        Returns
        -------
        pl.DataFrame
            Processed signals in the same long format.
        """
        if self.config.is_passthrough:
            return df

        wide = df.pivot(on="symbol", index="ts", values="signal").sort("ts")
        processed = self.process_wide(wide)
        return processed.unpivot(
            index="ts", variable_name="symbol", value_name="signal",
        ).drop_nulls("signal")


def _normalize_cross_sectional(
    df: pl.DataFrame,
    ts_col: str,
    signal_cols: list[str],
) -> pl.DataFrame:
    """Z-score normalize signal values cross-sectionally at each timestep."""
    row_mean = df.select(signal_cols).mean_horizontal()
    row_list = pl.concat_list([pl.col(c) for c in signal_cols])
    row_std = row_list.list.eval(pl.element().std(ddof=1)).list.first()
    safe_std = pl.when(row_std > 1e-12).then(row_std).otherwise(1.0)

    normed_exprs = [
        ((pl.col(c) - row_mean) / safe_std).alias(c)
        for c in signal_cols
    ]
    return df.select(pl.col(ts_col), *normed_exprs)


def _winsorize(
    df: pl.DataFrame,
    signal_cols: list[str],
    threshold: float,
) -> pl.DataFrame:
    """Clip signal values beyond +/- threshold standard deviations."""
    return df.with_columns(
        pl.col(c).clip(-threshold, threshold).alias(c)
        for c in signal_cols
    )


def _ema_smooth(
    df: pl.DataFrame,
    signal_cols: list[str],
    halflife: float,
) -> pl.DataFrame:
    """Apply exponential moving average smoothing per symbol column."""
    alpha = 1.0 - math.exp(-math.log(2.0) / halflife)
    span_equiv = (2.0 / alpha) - 1.0

    return df.with_columns(
        pl.col(c).ewm_mean(span=span_equiv, ignore_nulls=True, min_samples=1).alias(c)
        for c in signal_cols
    )
