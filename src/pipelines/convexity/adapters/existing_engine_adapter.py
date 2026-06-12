"""Adapter: existing `simple_backtest` engine -> convexity-pipeline `BacktestResult`.

The existing engine (`scripts/research/common/backtest.py::simple_backtest`) returns
portfolio-level series only. This adapter owns everything the convexity runner
needs on top of that:

* per-instrument loop + equal-weight aggregation,
* trade extraction (contiguous same-sign held-position runs),
* anchored walk-forward fold slicing (IS = anchor block, OOS = subsequent blocks),
* regime masking (bull / bear / sideways / high_vol / low_vol on the aggregate
  underlying),
* parameter perturbation enumeration (+/-20% band),
* universe-drop (remove the single largest net contributor),
* cost variants (simplified / pre-cost / realistic / 2x).

Signal contract
---------------
``candidate.signal_fn(bars, **params) -> pd.Series`` of *target positions*
(float, e.g. long/flat in {0, 1} or long/short in {-1, 0, 1}) aligned to
``bars.index``. The position decided on bar ``t`` is held from ``t+1``
(``execution_lag=1`` inside the engine) -> no look-ahead.

Variant approximations (documented per the implementation plan)
--------------------------------------------------------------
* Regime / fold variants slice the full per-bar net-return series and re-extract
  trades from the held-position runs *within the slice*. A trade that straddles a
  regime/fold boundary is therefore split at the boundary. This is the
  "regime-filtered trade slicing" approximation; it is conservative (it can only
  shorten trades, never invent them).
* Parameter perturbation scales every numeric parameter by a common factor drawn
  from a fixed +/-20% grid; integer params are rounded and floored at 1.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..types import BacktestResult, Candidate, CostModel
from .data_provider import LakeDataProvider, canonical_freq

# Import the existing engine. `backtest.py` uses a package-relative import
# (`from .data import ...`), so it must be loaded as part of its package via the
# repo-root namespace package path rather than as a bare top-level module.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.research.common.backtest import simple_backtest  # noqa: E402


# Fixed +/-20% perturbation grid (excludes 1.0 so every perturbation actually moves).
_PERTURB_FACTORS: Tuple[float, ...] = (0.80, 0.85, 0.90, 0.95, 1.05, 1.10, 1.15, 1.20)

# Default realistic round-trip costs (bps) by asset class.
_DEFAULT_REALISTIC_BPS: Dict[str, float] = {"crypto": 12.0, "etf": 4.0, "futures": 2.0}
_DEFAULT_SIMPLE_BPS = 5.0   # Stage-1 simplified cost (single number, all assets)


def _asset_class(symbol: str) -> str:
    if symbol.upper() in ("CL", "NG", "SI", "GC", "HG", "HO", "RB"):
        return "futures"
    if "-" in symbol:
        return "crypto"
    return "etf"


@dataclass
class ExistingEngineAdapter:
    """Wraps `simple_backtest` and emits `BacktestResult` for every runner variant."""

    provider: LakeDataProvider
    cost_model: CostModel = field(default_factory=CostModel)
    simple_bps: float = _DEFAULT_SIMPLE_BPS
    realistic_bps_by_class: Dict[str, float] = field(
        default_factory=lambda: dict(_DEFAULT_REALISTIC_BPS)
    )
    oos_fold_count: int = 8
    regime_lookback: int = 60
    execution_lag: int = 1
    _panel_cache: Dict[tuple, pd.DataFrame] = field(default_factory=dict, repr=False)

    # ------------------------------------------------------------------
    # Public entry point used by the runner: backtest_fn(candidate, variant)
    # ------------------------------------------------------------------
    def __call__(self, candidate: Candidate, variant: str) -> BacktestResult:
        return self.run(candidate, variant)

    def backtest_fn(self):
        """Return a `(candidate, variant) -> BacktestResult` closure for the runner."""
        return self.run

    def run(self, candidate: Candidate, variant: str) -> BacktestResult:
        freq = candidate.hypothesis.bar_frequency
        universe = list(candidate.hypothesis.universe)
        base_params = dict(candidate.hypothesis.params)
        base_params.update(candidate.backtest_kwargs.get("params", {}))

        # --- params (perturbation) ---
        params = base_params
        if variant.startswith("perturb_"):
            idx = int(variant.split("_")[-1])
            params = _perturb_params(base_params, _PERTURB_FACTORS[idx % len(_PERTURB_FACTORS)])

        # --- universe (drop largest contributor) ---
        if variant == "universe_drop" and len(universe) > 1:
            universe = self._universe_minus_top(candidate, universe, base_params)

        # --- cost (bps) ---
        cost_bps_map = self._cost_bps_for_variant(variant, universe)

        # --- per-instrument full results ---
        per_full: Dict[str, Dict[str, pd.Series]] = {}
        for sym in universe:
            full = self._full_instrument(candidate, sym, freq, params, cost_bps_map[sym])
            if full is not None and len(full["net_ret"]) > 0:
                per_full[sym] = full
        if not per_full:
            return _empty_result(meta={"variant": variant, "reason": "no_instrument_data"})

        # --- timestamp subset for this variant ---
        subset = self._subset_index(candidate, per_full, variant)

        # --- build per-instrument BacktestResults on the subset ---
        per_instrument: Dict[str, BacktestResult] = {}
        for sym, full in per_full.items():
            per_instrument[sym] = _slice_to_result(full, subset)

        agg = _aggregate(per_instrument, subset)
        agg.meta.update({"variant": variant, "universe": universe,
                         "cost_bps": cost_bps_map, "params": params})
        agg.per_instrument = per_instrument
        return agg

    # ------------------------------------------------------------------
    # Cost handling
    # ------------------------------------------------------------------
    def _cost_bps_for_variant(self, variant: str, universe: List[str]) -> Dict[str, float]:
        def realistic(sym: str) -> float:
            return self.realistic_bps_by_class.get(_asset_class(sym), 10.0)

        out: Dict[str, float] = {}
        for sym in universe:
            if variant in ("stage1_pre_cost", "stage2_pre_cost"):
                out[sym] = 0.0
            elif variant == "stage1_simple_cost":
                out[sym] = self.simple_bps
            elif variant == "cost_2x":
                out[sym] = 2.0 * realistic(sym)
            else:
                # stage2_realistic, is, oos_fold_*, perturb_*, regime_*, universe_drop
                out[sym] = realistic(sym)
        return out

    # ------------------------------------------------------------------
    # Instrument panel + full backtest (cached)
    # ------------------------------------------------------------------
    def _panel(self, candidate: Candidate, symbol: str, freq: str,
               params: Dict[str, Any]) -> Optional[pd.DataFrame]:
        key = (candidate.registry_id, symbol, canonical_freq(freq), _params_key(params))
        if key in self._panel_cache:
            return self._panel_cache[key]
        try:
            bars = self.provider.get_bars(symbol, freq)
        except Exception:
            self._panel_cache[key] = None
            return None
        if bars.empty or "close" not in bars:
            self._panel_cache[key] = None
            return None
        pos = candidate.signal_fn(bars, **params)
        pos = pd.Series(pos, index=bars.index).astype(float).fillna(0.0)
        under_ret = bars["close"].pct_change()
        panel = pd.DataFrame({"pos": pos, "under_ret": under_ret}).dropna(subset=["under_ret"])
        self._panel_cache[key] = panel
        return panel

    def _full_instrument(self, candidate: Candidate, symbol: str, freq: str,
                         params: Dict[str, Any], cost_bps: float
                         ) -> Optional[Dict[str, pd.Series]]:
        panel = self._panel(candidate, symbol, freq, params)
        if panel is None or panel.empty:
            return None
        weights = panel[["pos"]].rename(columns={"pos": symbol})
        returns = panel[["under_ret"]].rename(columns={"under_ret": symbol})
        bt = simple_backtest(weights, returns, cost_bps=cost_bps,
                             execution_lag=self.execution_lag)
        net_ret = pd.Series(bt["portfolio_ret"].values, index=pd.DatetimeIndex(bt["ts"]))
        held_pos = panel["pos"].shift(self.execution_lag).reindex(net_ret.index).fillna(0.0)
        under = panel["under_ret"].reindex(net_ret.index)
        return {"net_ret": net_ret, "under_ret": under, "held_pos": held_pos}

    def _universe_minus_top(self, candidate: Candidate, universe: List[str],
                            params: Dict[str, Any]) -> List[str]:
        freq = candidate.hypothesis.bar_frequency
        totals: Dict[str, float] = {}
        for sym in universe:
            full = self._full_instrument(
                candidate, sym, freq, params,
                self.realistic_bps_by_class.get(_asset_class(sym), 10.0),
            )
            totals[sym] = float(full["net_ret"].sum()) if full else -np.inf
        top = max(totals, key=totals.get)
        remaining = [s for s in universe if s != top]
        return remaining or universe

    # ------------------------------------------------------------------
    # Variant -> timestamp subset
    # ------------------------------------------------------------------
    def _full_union_index(self, per_full: Dict[str, Dict[str, pd.Series]]) -> pd.DatetimeIndex:
        idx: Optional[pd.DatetimeIndex] = None
        for full in per_full.values():
            i = full["net_ret"].index
            idx = i if idx is None else idx.union(i)
        return idx if idx is not None else pd.DatetimeIndex([])

    def _subset_index(self, candidate: Candidate,
                      per_full: Dict[str, Dict[str, pd.Series]],
                      variant: str) -> pd.DatetimeIndex:
        union = self._full_union_index(per_full)
        if len(union) == 0:
            return union

        if variant == "is":
            blocks = _split_blocks(union, self.oos_fold_count + 1)
            return blocks[0]
        if variant.startswith("oos_fold_"):
            i = int(variant.split("_")[-1])
            blocks = _split_blocks(union, self.oos_fold_count + 1)
            return blocks[i + 1] if i + 1 < len(blocks) else pd.DatetimeIndex([])
        if variant.startswith("regime_"):
            name = variant[len("regime_"):]
            return self._regime_mask(per_full, union, name)
        # all other variants use the full sample
        return union

    def _regime_mask(self, per_full: Dict[str, Dict[str, pd.Series]],
                     union: pd.DatetimeIndex, name: str) -> pd.DatetimeIndex:
        # Aggregate underlying = equal-weight mean across instruments present per bar.
        und = pd.concat(
            [full["under_ret"].reindex(union) for full in per_full.values()], axis=1
        ).mean(axis=1)
        mom = und.rolling(self.regime_lookback, min_periods=max(5, self.regime_lookback // 3)).sum()
        vol = und.rolling(self.regime_lookback, min_periods=max(5, self.regime_lookback // 3)).std()
        vol_med = vol.median()
        if name == "bull":
            mask = mom > 0
        elif name == "bear":
            mask = mom < 0
        elif name == "sideways":
            band = mom.abs().quantile(0.25)
            mask = mom.abs() <= band
        elif name == "high_vol":
            mask = vol > vol_med
        elif name == "low_vol":
            mask = vol <= vol_med
        else:
            mask = pd.Series(True, index=union)
        return union[mask.reindex(union).fillna(False).values]


# ----------------------------------------------------------------------
# Module-level helpers
# ----------------------------------------------------------------------
def _params_key(params: Dict[str, Any]) -> tuple:
    return tuple(sorted((k, round(v, 8) if isinstance(v, float) else v)
                        for k, v in params.items()))


def _perturb_params(params: Dict[str, Any], factor: float) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in params.items():
        if isinstance(v, bool):
            out[k] = v
        elif isinstance(v, int):
            out[k] = max(1, int(round(v * factor)))
        elif isinstance(v, float):
            out[k] = v * factor
        else:
            out[k] = v
    return out


def _split_blocks(index: pd.DatetimeIndex, n_blocks: int) -> List[pd.DatetimeIndex]:
    """Split a sorted DatetimeIndex into ``n_blocks`` contiguous equal-size blocks."""
    if n_blocks <= 0 or len(index) == 0:
        return [index]
    bounds = np.linspace(0, len(index), n_blocks + 1, dtype=int)
    return [index[bounds[i]:bounds[i + 1]] for i in range(n_blocks)]


def _extract_trades(net_ret: pd.Series, held_pos: pd.Series
                    ) -> Tuple[pd.Series, pd.Series]:
    """Extract per-trade PnL (sum of net returns) and duration (bars) from runs."""
    df = pd.DataFrame({"sign": np.sign(held_pos), "ret": net_ret}).dropna(subset=["ret"])
    if df.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    grp = (df["sign"] != df["sign"].shift()).cumsum()
    pnls: List[float] = []
    durs: List[float] = []
    for _, g in df.groupby(grp):
        if g["sign"].iloc[0] == 0:
            continue
        pnls.append(float(g["ret"].sum()))
        durs.append(float(len(g)))
    return (pd.Series(pnls, dtype=float), pd.Series(durs, dtype=float))


def _slice_to_result(full: Dict[str, pd.Series], subset: pd.DatetimeIndex) -> BacktestResult:
    net = full["net_ret"].reindex(subset).dropna()
    under = full["under_ret"].reindex(net.index)
    held = full["held_pos"].reindex(net.index).fillna(0.0)
    equity = (1.0 + net).cumprod()
    pnls, durs = _extract_trades(net, held)
    return BacktestResult(
        alpha_returns=net,
        underlying_returns=under,
        equity=equity,
        trade_pnls=pnls,
        trade_durations=durs,
    )


def _aggregate(per_instrument: Dict[str, BacktestResult],
               subset: pd.DatetimeIndex) -> BacktestResult:
    """Equal-weight aggregate of per-instrument results; pool trades."""
    alpha_mat = pd.concat(
        [bt.alpha_returns.reindex(subset) for bt in per_instrument.values()], axis=1
    )
    und_mat = pd.concat(
        [bt.underlying_returns.reindex(subset) for bt in per_instrument.values()], axis=1
    )
    alpha = alpha_mat.mean(axis=1).dropna()
    under = und_mat.mean(axis=1).reindex(alpha.index)
    equity = (1.0 + alpha).cumprod()
    pnls = pd.concat([bt.trade_pnls for bt in per_instrument.values()],
                     ignore_index=True) if per_instrument else pd.Series(dtype=float)
    durs = pd.concat([bt.trade_durations for bt in per_instrument.values()],
                     ignore_index=True) if per_instrument else pd.Series(dtype=float)
    return BacktestResult(
        alpha_returns=alpha,
        underlying_returns=under,
        equity=equity,
        trade_pnls=pnls.reset_index(drop=True),
        trade_durations=durs.reset_index(drop=True),
        meta={"n_instruments": len(per_instrument)},
    )


def _empty_result(meta: Optional[Dict[str, Any]] = None) -> BacktestResult:
    empty = pd.Series(dtype=float)
    return BacktestResult(
        alpha_returns=empty, underlying_returns=empty, equity=empty,
        trade_pnls=empty, trade_durations=empty, meta=meta or {},
    )
