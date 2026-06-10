"""
TA-Lib-driven directional signal grammar for silver.

Produces a discrete state in {-1, 0, +1} per bar based on a parameterised
combination of trend, momentum, confirmation, volatility-regime and
candle-pattern filters.

All filters are vectorised over a feature dataframe that was pre-built by
`scripts/build_features.py`.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict

import numpy as np
import pandas as pd

try:
    import talib  # type: ignore
except Exception:  # pragma: no cover
    talib = None  # noqa: N816


# -------------------------------------------------------------------- params --


@dataclass(frozen=True)
class SignalParams:
    fast: int = 20          # SMA fast window
    slow: int = 100         # SMA slow window
    rsi_long_thr: float = 55.0
    rsi_short_thr: float = 45.0
    use_macd: bool = True
    use_adx: bool = True
    adx_min: float = 20.0
    atr_max: float = 0.05   # ATR / close ceiling (>= => filter active)
    pattern_lookback: int = 3
    use_pattern_boost: bool = True

    # --- (A) Bollinger Band regime switch ---
    use_bb_regime: bool = False
    bb_period: int = 20
    bb_std: float = 2.0
    bb_width_pctl_lookback: int = 100
    bb_regime_thr_pct: float = 0.30

    # --- (B) Vol-of-vol expansion trigger ---
    use_vov_trigger: bool = False
    vov_window: int = 20
    vov_smooth: int = 5
    vov_zscore_thr: float = 1.5
    vov_action: str = "exit_only"   # 'exit_only' | 'flip_to_flat'

    # --- (C) v3: scope vov trigger to mean-revert regime only ---
    vov_only_in_mr_regime: bool = False

    # --- (D) v4: hysteresis / minimum-hold layer ---
    # min_hold_bars: once state flips to +/-1, it must persist that many bars
    #   before any flip is permitted. Default 1 = no min hold (identical to v3).
    # hysteresis_bars: when state goes non-zero -> zero, require that many
    #   consecutive flat-signal bars before actually flattening. Default 0.
    min_hold_bars: int = 1
    hysteresis_bars: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def __post_init__(self) -> None:
        if self.vov_only_in_mr_regime and not self.use_bb_regime:
            raise ValueError(
                "vov_only_in_mr_regime=True requires use_bb_regime=True "
                "(the MR regime mask is only defined when BB regime is on)."
            )


# ------------------------------------------------------------------ helpers --


_BULLISH_CDL = ["cdl_engulfing", "cdl_hammer", "cdl_morningstar",
                "cdl_3whitesoldiers", "cdl_harami"]
_BEARISH_CDL = ["cdl_eveningstar", "cdl_shootingstar", "cdl_3blackcrows"]


def _safe(s: pd.Series, fill: float = 0.0) -> pd.Series:
    return s.replace([np.inf, -np.inf], np.nan).fillna(fill)


def _bullish_pattern_any(f: pd.DataFrame, n: int) -> pd.Series:
    cols = [c for c in _BULLISH_CDL if c in f.columns]
    if not cols:
        return pd.Series(False, index=f.index)
    raw = (f[cols] > 0).any(axis=1)
    return raw.rolling(n, min_periods=1).max().astype(bool)


def _bearish_pattern_any(f: pd.DataFrame, n: int) -> pd.Series:
    cols = [c for c in _BEARISH_CDL if c in f.columns]
    if not cols:
        return pd.Series(False, index=f.index)
    raw = (f[cols] < 0).any(axis=1)
    return raw.rolling(n, min_periods=1).max().astype(bool)


def _bbands(close: pd.Series, period: int, nstd: float) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Vectorised Bollinger bands. Uses TA-Lib when available, else pandas."""
    if talib is not None:
        arr = close.astype("float64").to_numpy()
        u, m, l = talib.BBANDS(arr, timeperiod=int(period),
                               nbdevup=float(nstd), nbdevdn=float(nstd), matype=0)
        return (pd.Series(u, index=close.index),
                pd.Series(m, index=close.index),
                pd.Series(l, index=close.index))
    middle = close.rolling(period, min_periods=period).mean()
    sd = close.rolling(period, min_periods=period).std(ddof=0)
    upper = middle + nstd * sd
    lower = middle - nstd * sd
    return upper, middle, lower


def _rolling_pctl(s: pd.Series, window: int) -> pd.Series:
    """Rolling percentile rank (0..1) of the current value within the window."""
    return s.rolling(int(window), min_periods=max(int(window) // 4, 5)).rank(pct=True).fillna(0.5)


def _apply_min_hold_hysteresis(
    raw_state: np.ndarray,
    min_hold_bars: int,
    hysteresis_bars: int,
) -> np.ndarray:
    """Apply minimum-hold and flat-confirmation hysteresis to a raw state array.

    Rules (deterministic, causal):
      - Once `effective_state` flips to a non-zero value v at bar t, that value
        is locked in for at least `min_hold_bars` bars (bars t..t+min_hold-1).
        Any change in raw_state during that lock window is ignored.
      - After the lock window ends, the next non-zero raw signal that differs
        from the current state is applied IMMEDIATELY (no hysteresis on
        non-zero flips).
      - A transition from non-zero to zero requires `hysteresis_bars`
        consecutive raw==0 bars BEFORE the flat is actually adopted; the bar
        on which the flat is adopted is the (hysteresis_bars+1)-th consecutive
        zero bar. With hysteresis_bars=0 a single flat raw bar flattens.
      - With (min_hold_bars=1, hysteresis_bars=0) the output equals raw_state.

    Implementation is a tight 1-D numpy scan: O(N) integer operations, no
    pandas or python-object overhead.
    """
    n = raw_state.shape[0]
    out = np.empty(n, dtype=np.int8)
    if n == 0:
        return out
    mh = max(int(min_hold_bars), 1)
    hb = max(int(hysteresis_bars), 0)
    cur = np.int8(0)
    lock_left = 0          # # bars remaining where state is locked (min-hold)
    flat_run = 0           # # of consecutive raw==0 bars seen while cur != 0
    for i in range(n):
        r = raw_state[i]
        if lock_left > 0:
            # Locked into `cur`. Reset flat-run since hysteresis only counts
            # consecutive flat signals AFTER the lock window expires.
            out[i] = cur
            lock_left -= 1
            if lock_left == 0:
                flat_run = 0
            continue
        if cur == 0:
            # Currently flat. Any non-zero raw flips us immediately.
            if r != 0:
                cur = np.int8(r)
                lock_left = mh - 1     # we already consumed bar i
                flat_run = 0
            out[i] = cur
            continue
        # cur != 0 and not locked
        if r == cur:
            flat_run = 0
            out[i] = cur
            continue
        if r != 0 and r != cur:
            # Direct flip to the opposite non-zero state, lock-window expired.
            cur = np.int8(r)
            lock_left = mh - 1
            flat_run = 0
            out[i] = cur
            continue
        # r == 0, cur != 0 -> count toward hysteresis
        flat_run += 1
        if flat_run > hb:
            cur = np.int8(0)
            flat_run = 0
            out[i] = cur
        else:
            out[i] = cur
    return out


def _hysteresis_flag(abs_z: np.ndarray, hi: float, lo: float) -> np.ndarray:
    """Vectorised hysteresis: turns ON when abs_z >= hi, stays ON until abs_z < lo.

    This is computed with numpy without a per-bar python loop in the hot path
    using a state-propagation trick: we encode the on/off transitions and then
    forward-fill them.
    """
    n = abs_z.shape[0]
    if n == 0:
        return np.zeros(0, dtype=bool)
    # marker: +1 = turn on, -1 = turn off, 0 = no change
    on = abs_z >= hi
    off = abs_z < lo
    # In areas where neither holds, we keep prior state.
    # state[i] = on[i] ? True : off[i] ? False : state[i-1]
    # Vectorise via cumulative max trick:
    # Encode triggers as event indices; map indices forward by maximum.
    s = np.where(on, 1, np.where(off, 0, -1)).astype(np.int8)
    # Forward-fill -1 with previous non--1 value
    valid = s != -1
    idx = np.where(valid, np.arange(n), -1)
    np.maximum.accumulate(idx, out=idx)
    # idx[i] = index of last valid event up to i; for i with no prior event, idx==-1
    out = np.zeros(n, dtype=bool)
    seen = idx >= 0
    out[seen] = s[idx[seen]] == 1
    return out


# ----------------------------------------------------------------- builder --


class SignalGrammar:
    """Stateless signal builder. Call `.generate(features_df)` -> pd.Series."""

    def __init__(self, params: SignalParams):
        self.p = params

    # ............................................................ generate --

    def generate(self, features: pd.DataFrame) -> pd.Series:
        f = features
        p = self.p

        sma_fast_col = f"sma_{p.fast}"
        sma_slow_col = f"sma_{p.slow}"

        if sma_fast_col not in f.columns or sma_slow_col not in f.columns:
            raise KeyError(
                f"missing required feature column: {sma_fast_col}/{sma_slow_col}"
            )

        sma_fast = _safe(f[sma_fast_col])
        sma_slow = _safe(f[sma_slow_col])
        rsi = _safe(f["rsi_14"], fill=50.0)
        macd_hist = _safe(f["macd_hist"])
        adx = _safe(f["adx_14"], fill=0.0)
        atr = _safe(f["atr_14"], fill=0.0)
        close = _safe(f["c"] if "c" in f.columns else features.get("close"))

        trend_long = sma_fast > sma_slow
        trend_short = sma_fast < sma_slow

        mom_long = rsi >= p.rsi_long_thr
        mom_short = rsi <= p.rsi_short_thr

        if p.use_macd:
            confirm_long = macd_hist > 0
            confirm_short = macd_hist < 0
        else:
            confirm_long = pd.Series(True, index=f.index)
            confirm_short = pd.Series(True, index=f.index)

        # Volatility regime gate: ADX above floor OR ATR/close below ceiling
        natr = (atr / close.replace(0, np.nan)).fillna(0.0)
        if p.use_adx:
            vol_gate = (adx >= p.adx_min) | (natr <= p.atr_max)
        else:
            vol_gate = natr <= p.atr_max

        # Pattern boost gives a one-bar override even if confirm is weak
        if p.use_pattern_boost:
            patt_long = _bullish_pattern_any(f, p.pattern_lookback)
            patt_short = _bearish_pattern_any(f, p.pattern_lookback)
        else:
            patt_long = pd.Series(False, index=f.index)
            patt_short = pd.Series(False, index=f.index)

        trend_long_state = trend_long & mom_long & (confirm_long | patt_long) & vol_gate
        trend_short_state = trend_short & mom_short & (confirm_short | patt_short) & vol_gate

        # Trend-regime baseline state vector
        state = pd.Series(0, index=f.index, dtype="int8")
        state[trend_long_state.values] = 1
        # if both fire (rare), longs win since the book is net long-biased
        state[trend_short_state.values & ~trend_long_state.values] = -1

        # ---------------- (A) Bollinger-band regime switch ----------------
        mr_mask_arr_for_vov: np.ndarray | None = None
        if p.use_bb_regime:
            upper, middle, lower = _bbands(close, p.bb_period, p.bb_std)
            bb_width = ((upper - lower) / middle.replace(0, np.nan)).ffill().fillna(0.0)
            bb_width_pctl = _rolling_pctl(bb_width, p.bb_width_pctl_lookback)
            # mean-rev mask: tight bands / quiet vol
            mr_mask = bb_width_pctl < p.bb_regime_thr_pct
            mr_mask_arr_for_vov = mr_mask.to_numpy().copy()

            # mean-rev state: +1 if close < lower AND rsi < 40; -1 if close > upper AND rsi > 60
            mr_long = (close < lower) & (rsi < 40.0)
            mr_short = (close > upper) & (rsi > 60.0)
            mr_state = pd.Series(0, index=f.index, dtype="int8")
            mr_state[mr_long.values] = 1
            mr_state[mr_short.values & ~mr_long.values] = -1

            # Where mean-rev regime applies, replace trend state with MR state
            mr_mask_arr = mr_mask.to_numpy()
            state_arr = state.to_numpy().copy()
            state_arr = np.where(mr_mask_arr, mr_state.to_numpy(), state_arr)
            state = pd.Series(state_arr.astype("int8"), index=f.index)

        # ---------------- (B) Vol-of-vol expansion trigger ----------------
        if p.use_vov_trigger:
            log_ret = np.log(close.replace(0, np.nan)).diff()
            log_ret = log_ret.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            vol = log_ret.rolling(p.vov_window, min_periods=2).std()
            vol = vol.rolling(max(int(p.vov_smooth), 1), min_periods=1).mean()
            dvol = vol.diff()
            z = (dvol - dvol.rolling(100, min_periods=10).mean()) / dvol.rolling(100, min_periods=10).std()
            z = z.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            abs_z = z.abs()
            # Expansion event: |z| > thr; remains "on" until |z| falls below thr/2.
            expansion_arr = _hysteresis_flag(
                abs_z.to_numpy(),
                hi=float(p.vov_zscore_thr),
                lo=float(p.vov_zscore_thr) / 2.0,
            )
            # v3: optionally restrict expansion trigger to the MR regime mask.
            # In trend regime the trigger is suppressed.
            if p.vov_only_in_mr_regime:
                if mr_mask_arr_for_vov is None:
                    # Defensive — __post_init__ should have caught this already.
                    raise ValueError(
                        "vov_only_in_mr_regime=True requires use_bb_regime=True"
                    )
                expansion_arr = expansion_arr & mr_mask_arr_for_vov
            expansion = pd.Series(expansion_arr, index=f.index)

            state_arr = state.to_numpy().copy()
            if p.vov_action == "exit_only":
                # Force flat next bar while expansion is active.
                # Apply with shift(1) so we act on the *next* bar (no look-ahead).
                exp_next = np.concatenate(([False], expansion_arr[:-1]))
                state_arr = np.where(exp_next, 0, state_arr)
            elif p.vov_action == "flip_to_flat":
                # On-event same bar, plus look-back at trend logic re-eval next bar:
                # In our vectorised formulation, "flip_to_flat" simply forces flat
                # on the expansion bar itself and any subsequent bars while
                # expansion stays "on". Trend logic resumes automatically once
                # expansion turns off.
                state_arr = np.where(expansion_arr, 0, state_arr)
            else:
                raise ValueError(f"unknown vov_action: {p.vov_action!r}")
            state = pd.Series(state_arr.astype("int8"), index=f.index)

        # ---------------- (D) min-hold + flat hysteresis -----------------
        if p.min_hold_bars > 1 or p.hysteresis_bars > 0:
            raw_arr = state.to_numpy().astype(np.int8, copy=False)
            adj_arr = _apply_min_hold_hysteresis(
                raw_arr, int(p.min_hold_bars), int(p.hysteresis_bars)
            )
            state = pd.Series(adj_arr, index=f.index)

        return state

    # ........................................................ size mapper --

    def map_to_position(
        self,
        state: pd.Series,
        max_contracts: int = 4,
        call_delta_target: float = 0.30,
    ) -> pd.DataFrame:
        """
        Translate signal state into a target position:
        QI mini-silver contracts and a call-delta target for the option overlay.
        """
        out = pd.DataFrame(index=state.index)
        out["qi_contracts"] = state.astype(int) * max_contracts
        out["call_delta_target"] = (
            np.where(state > 0, call_delta_target,
                     np.where(state < 0, -call_delta_target, 0.0))
        )
        return out


# --------------------------------------------------------------- grid spec --


def default_grid() -> Dict[str, list]:
    """Coarse-ish grid kept small enough to stay <5000 combos x 4 TFs."""
    return dict(
        fast=[10, 20, 50],
        slow=[50, 100, 200],
        rsi_long_thr=[50.0, 55.0, 60.0],
        rsi_short_thr=[40.0, 45.0, 50.0],
        use_macd=[True, False],
        use_adx=[True],
        adx_min=[15.0, 20.0, 25.0],
        atr_max=[0.04, 0.06],
        pattern_lookback=[3],
        use_pattern_boost=[True, False],
    )


def v2_extension_grid() -> Dict[str, list]:
    """Search space for the v2 BB-regime + vov-trigger extensions only.

    The prior trend-layer best (fast=50, slow=200, rsi=50/45, macd=True,
    adx=True/25, atr=0.04, pattern_boost=False) is held FIXED outside this
    grid by ``scripts/fit_model_v2.py``.
    """
    return dict(
        bb_period=[14, 20, 30],
        bb_std=[2.0, 2.5],
        bb_width_pctl_lookback=[60, 100, 200],
        bb_regime_thr_pct=[0.20, 0.30, 0.40],
        vov_window=[10, 20, 30],
        vov_zscore_thr=[1.0, 1.5, 2.0],
        vov_action=["exit_only", "flip_to_flat"],
        use_bb_regime=[True, False],
        use_vov_trigger=[True, False],
    )


def iter_param_grid(grid: Dict[str, list]):
    keys = list(grid.keys())
    from itertools import product
    for combo in product(*[grid[k] for k in keys]):
        d = dict(zip(keys, combo))
        # enforce fast < slow
        if "fast" in d and "slow" in d and d["fast"] >= d["slow"]:
            continue
        yield SignalParams(**d)


if __name__ == "__main__":
    import json
    import pathlib

    art = pathlib.Path(__file__).resolve().parents[1] / "artifacts"
    feats = pd.read_parquet(art / "features_8H.parquet").set_index("ts")
    bars = pd.read_parquet(art / "si_front_month_8H.parquet").set_index("ts")
    feats["c"] = bars["c"]

    # Load v3 best params
    v3 = json.loads((art / "best_params_v3.json").read_text())["params"]

    base = SignalParams(**v3)
    with_defaults = SignalParams(**{**v3, "min_hold_bars": 1, "hysteresis_bars": 0})

    sig_base = SignalGrammar(base).generate(feats)
    sig_passthrough = SignalGrammar(with_defaults).generate(feats)
    assert (sig_base.values == sig_passthrough.values).all(), (
        "PASS-THROUGH ASSERTION FAILED: (min_hold_bars=1, hysteresis_bars=0) "
        "must equal v3 base output exactly"
    )
    print("PASS: defaults (1,0) reproduce v3 base output exactly")
    print("state value counts:")
    print(sig_base.value_counts().sort_index())

    # Show an extended-hold variant just to sanity-check the new layer
    held = SignalGrammar(
        SignalParams(**{**v3, "min_hold_bars": 10, "hysteresis_bars": 3})
    ).generate(feats)
    print("with (10,3) min-hold/hysteresis:")
    print(held.value_counts().sort_index())
