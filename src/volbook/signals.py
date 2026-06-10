"""Derive ranked technical risk:reward setups for a bar series.

The indicator tables tell you *where* every TA-Lib function sits; this
module turns that into a small, curated list of *actionable* long/short
setups so the canvas can show "if I had to take a trade right now, here
are the best risk:reward ideas given the data."

Each setup is evaluated against canonical indicator confluence rules
(trend continuation, Bollinger mean-reversion, range breakout). A
setup is emitted only when its preconditions fire; entries, stops, and
targets are anchored to ATR (and BBANDS or range levels where it makes
sense) so the reward/risk ratio is comparable across ideas.

The returned records are plain dicts so they serialise straight into
the canvas bundle.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

try:
    import talib

    TALIB_AVAILABLE = True
except ImportError:  # pragma: no cover
    TALIB_AVAILABLE = False


@dataclass
class TradeSetup:
    name: str
    direction: str  # "long" | "short"
    entry: float
    stop: float
    target: float
    risk: float
    reward: float
    rr: float
    confidence: float  # 0..1 — heuristic, for ranking only
    score: float  # rr * confidence (used for final ordering)
    rationale: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class RegimeState:
    """Strategy-class gate computed before setup scoring."""

    name: str
    adx: float
    eligible_tags: frozenset[str]
    rationale: str


def _finite(x: Any) -> bool:
    if x is None:
        return False
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return False
    return True


def classify_regime(
    bars: list[Any],
    *,
    adx_mean_reversion_max: float = 20.0,
    adx_trend_min: float = 25.0,
) -> RegimeState:
    """Classify the latest bar into strategy eligibility buckets.

    Assumption: until long-history validation exists, ADX is the baseline
    production gate. Low ADX admits mean reversion, high ADX admits trend
    and breakout, and the middle band is a no-trade transition zone.
    """
    if not TALIB_AVAILABLE or len(bars) < 60:
        return RegimeState("insufficient", float("nan"), frozenset(), "insufficient history")

    h = np.array([b.h for b in bars], dtype=np.float64)
    l = np.array([b.l for b in bars], dtype=np.float64)
    c = np.array([b.c for b in bars], dtype=np.float64)
    adx = float(talib.ADX(h, l, c, timeperiod=14)[-1])
    if not _finite(adx):
        return RegimeState("unknown", adx, frozenset(), "ADX unavailable")
    if adx <= adx_mean_reversion_max:
        return RegimeState(
            "mean_reversion",
            adx,
            frozenset({"mean-reversion"}),
            f"ADX {adx:.1f} <= {adx_mean_reversion_max:.1f}",
        )
    if adx >= adx_trend_min:
        return RegimeState(
            "trend",
            adx,
            frozenset({"trend", "breakout"}),
            f"ADX {adx:.1f} >= {adx_trend_min:.1f}",
        )
    return RegimeState(
        "transition",
        adx,
        frozenset(),
        f"ADX {adx:.1f} in no-trade band {adx_mean_reversion_max:.1f}-{adx_trend_min:.1f}",
    )


def build_setups(bars: list[Any], *, max_setups: int = 6) -> list[dict]:
    """Build a ranked list of actionable technical setups.

    Returns a list of dicts (already JSON-friendly). Empty when TA-Lib
    isn't available, when there aren't enough bars to evaluate the
    longest-lookback indicator, or when no candidate setup clears the
    RR≥1 floor.
    """
    if not TALIB_AVAILABLE or len(bars) < 60:
        return []
    regime = classify_regime(bars)
    if not regime.eligible_tags:
        return []

    o = np.array([b.o for b in bars], dtype=np.float64)
    h = np.array([b.h for b in bars], dtype=np.float64)
    l = np.array([b.l for b in bars], dtype=np.float64)
    c = np.array([b.c for b in bars], dtype=np.float64)
    v = np.array([b.v for b in bars], dtype=np.float64)

    # All indicator series are full-resolution; we read `[-1]` for the
    # latest value and `[-2]` where we care about direction (rising vs
    # falling histograms, etc.).
    atr = talib.ATR(h, l, c, timeperiod=14)[-1]
    rsi = talib.RSI(c, timeperiod=14)[-1]
    sma20 = talib.SMA(c, timeperiod=20)[-1]
    sma50 = talib.SMA(c, timeperiod=50)[-1]
    adx = talib.ADX(h, l, c, timeperiod=14)[-1]
    plus_di = talib.PLUS_DI(h, l, c, timeperiod=14)[-1]
    minus_di = talib.MINUS_DI(h, l, c, timeperiod=14)[-1]
    macd_arr, macdsig_arr, macdhist_arr = talib.MACD(c)
    macd = macd_arr[-1]
    macdsig = macdsig_arr[-1]
    hist = macdhist_arr[-1]
    hist_prev = macdhist_arr[-2] if len(macdhist_arr) >= 2 else float("nan")
    bb_u_arr, bb_m_arr, bb_l_arr = talib.BBANDS(
        c, timeperiod=20, nbdevup=2.0, nbdevdn=2.0
    )
    bb_u = bb_u_arr[-1]
    bb_m = bb_m_arr[-1]
    bb_l = bb_l_arr[-1]
    sar = talib.SAR(h, l, acceleration=0.02, maximum=0.2)[-1]
    stoch_k_arr, stoch_d_arr = talib.STOCH(h, l, c)
    stoch_k = stoch_k_arr[-1]
    stoch_d = stoch_d_arr[-1]

    # Range references for breakouts — the 20 bars *prior* to the
    # latest one so "close > high20" actually means a fresh high.
    if len(h) >= 21:
        high20 = float(h[-21:-1].max())
        low20 = float(l[-21:-1].min())
    else:
        high20 = float(h[:-1].max()) if len(h) > 1 else float(h[-1])
        low20 = float(l[:-1].min()) if len(l) > 1 else float(l[-1])

    close = float(c[-1])
    setups: list[TradeSetup] = []

    # ── 1. Trend continuation LONG ─────────────────────────────────
    if (
        _finite(atr)
        and _finite(sma20)
        and _finite(adx)
        and _finite(plus_di)
        and _finite(minus_di)
        and _finite(hist)
    ):
        if "trend" in regime.eligible_tags and close > sma20 and adx >= 18 and plus_di > minus_di and hist > 0:
            entry = close
            stop = entry - 2.0 * atr
            # Prefer SAR as a tighter stop if it sits just under price.
            if _finite(sar) and sar < close and sar > stop:
                stop = float(sar)
            target = entry + 3.0 * atr
            rationale = [
                f"Close {close:.2f} above SMA20 {sma20:.2f}",
                f"ADX {adx:.1f} with +DI {plus_di:.1f} > −DI {minus_di:.1f}",
                f"MACD hist {hist:+.2f}",
            ]
            conf = 0.40
            if _finite(sma50) and close > sma50:
                conf += 0.10
                rationale.append(f"Above SMA50 {sma50:.2f}")
            if adx >= 25:
                conf += 0.10
                rationale.append("ADX ≥ 25 (strong trend)")
            if _finite(hist_prev) and hist > hist_prev:
                conf += 0.10
                rationale.append("MACD histogram rising")
            if _finite(rsi) and 50 < rsi < 70:
                conf += 0.05
                rationale.append(f"RSI {rsi:.0f} bullish, not overbought")
            if _finite(sar) and sar < close:
                conf += 0.05
                rationale.append("Parabolic SAR trailing below price")
            risk = entry - stop
            reward = target - entry
            if risk > 0 and reward > 0:
                setups.append(
                    TradeSetup(
                        name="Trend continuation",
                        direction="long",
                        entry=entry,
                        stop=float(stop),
                        target=float(target),
                        risk=float(risk),
                        reward=float(reward),
                        rr=float(reward / risk),
                        confidence=min(conf, 0.95),
                        score=0.0,
                        rationale=[regime.rationale, *rationale],
                        tags=["trend", "long"],
                    )
                )

    # ── 2. Trend continuation SHORT ────────────────────────────────
    if (
        _finite(atr)
        and _finite(sma20)
        and _finite(adx)
        and _finite(plus_di)
        and _finite(minus_di)
        and _finite(hist)
    ):
        if "trend" in regime.eligible_tags and close < sma20 and adx >= 18 and minus_di > plus_di and hist < 0:
            entry = close
            stop = entry + 2.0 * atr
            if _finite(sar) and sar > close and sar < stop:
                stop = float(sar)
            target = entry - 3.0 * atr
            rationale = [
                f"Close {close:.2f} below SMA20 {sma20:.2f}",
                f"ADX {adx:.1f} with −DI {minus_di:.1f} > +DI {plus_di:.1f}",
                f"MACD hist {hist:+.2f}",
            ]
            conf = 0.40
            if _finite(sma50) and close < sma50:
                conf += 0.10
                rationale.append(f"Below SMA50 {sma50:.2f}")
            if adx >= 25:
                conf += 0.10
                rationale.append("ADX ≥ 25 (strong trend)")
            if _finite(hist_prev) and hist < hist_prev:
                conf += 0.10
                rationale.append("MACD histogram falling")
            if _finite(rsi) and 30 < rsi < 50:
                conf += 0.05
                rationale.append(f"RSI {rsi:.0f} bearish, not oversold")
            if _finite(sar) and sar > close:
                conf += 0.05
                rationale.append("Parabolic SAR trailing above price")
            risk = stop - entry
            reward = entry - target
            if risk > 0 and reward > 0:
                setups.append(
                    TradeSetup(
                        name="Trend continuation",
                        direction="short",
                        entry=entry,
                        stop=float(stop),
                        target=float(target),
                        risk=float(risk),
                        reward=float(reward),
                        rr=float(reward / risk),
                        confidence=min(conf, 0.95),
                        score=0.0,
                        rationale=[regime.rationale, *rationale],
                        tags=["trend", "short"],
                    )
                )

    # ── 3. Mean reversion LONG (Bollinger bounce) ──────────────────
    if _finite(atr) and _finite(rsi) and _finite(bb_l) and _finite(bb_m):
        if "mean-reversion" in regime.eligible_tags and rsi < 35 and close <= bb_l * 1.005 and bb_m > close:
            entry = close
            # Stop below the lower band or one ATR — whichever is wider.
            stop = min(bb_l * 0.99, entry - 1.5 * atr)
            target = bb_m
            rationale = [
                f"RSI {rsi:.0f} oversold",
                f"Close {close:.2f} at/below lower BB {bb_l:.2f}",
                f"Target middle BB {bb_m:.2f}",
            ]
            conf = 0.40
            if _finite(stoch_k) and stoch_k < 20:
                conf += 0.10
                rationale.append(f"Stoch %K {stoch_k:.0f} oversold")
            if rsi < 25:
                conf += 0.10
                rationale.append("RSI < 25 (deep oversold)")
            if _finite(adx) and adx < 25:
                conf += 0.05
                rationale.append("ADX < 25 (not a strong downtrend)")
            risk = entry - stop
            reward = target - entry
            if risk > 0 and reward > 0:
                setups.append(
                    TradeSetup(
                        name="Mean reversion (BB bounce)",
                        direction="long",
                        entry=entry,
                        stop=float(stop),
                        target=float(target),
                        risk=float(risk),
                        reward=float(reward),
                        rr=float(reward / risk),
                        confidence=min(conf, 0.90),
                        score=0.0,
                        rationale=[regime.rationale, *rationale],
                        tags=["mean-reversion", "long"],
                    )
                )

    # ── 4. Mean reversion SHORT (Bollinger fade) ───────────────────
    if _finite(atr) and _finite(rsi) and _finite(bb_u) and _finite(bb_m):
        if "mean-reversion" in regime.eligible_tags and rsi > 65 and close >= bb_u * 0.995 and bb_m < close:
            entry = close
            stop = max(bb_u * 1.01, entry + 1.5 * atr)
            target = bb_m
            rationale = [
                f"RSI {rsi:.0f} overbought",
                f"Close {close:.2f} at/above upper BB {bb_u:.2f}",
                f"Target middle BB {bb_m:.2f}",
            ]
            conf = 0.40
            if _finite(stoch_k) and stoch_k > 80:
                conf += 0.10
                rationale.append(f"Stoch %K {stoch_k:.0f} overbought")
            if rsi > 75:
                conf += 0.10
                rationale.append("RSI > 75 (deep overbought)")
            if _finite(adx) and adx < 25:
                conf += 0.05
                rationale.append("ADX < 25 (not a strong uptrend)")
            risk = stop - entry
            reward = entry - target
            if risk > 0 and reward > 0:
                setups.append(
                    TradeSetup(
                        name="Mean reversion (BB fade)",
                        direction="short",
                        entry=entry,
                        stop=float(stop),
                        target=float(target),
                        risk=float(risk),
                        reward=float(reward),
                        rr=float(reward / risk),
                        confidence=min(conf, 0.90),
                        score=0.0,
                        rationale=[regime.rationale, *rationale],
                        tags=["mean-reversion", "short"],
                    )
                )

    # ── 5. Range breakout LONG ─────────────────────────────────────
    if "breakout" in regime.eligible_tags and _finite(atr) and _finite(high20) and close > high20:
        entry = close
        stop = entry - 1.5 * atr
        range_size = high20 - low20
        target = entry + max(2.0 * atr, range_size)
        rationale = [
            f"Close {close:.2f} > prior 20-bar high {high20:.2f}",
            f"Project range {range_size:.2f} vs 2×ATR {2*atr:.2f}",
        ]
        conf = 0.40
        if _finite(adx) and adx >= 18:
            conf += 0.10
            rationale.append(f"ADX {adx:.1f} confirms momentum")
        if len(v) >= 21 and v[-21:-1].mean() > 0 and v[-1] > v[-21:-1].mean() * 1.2:
            conf += 0.10
            rationale.append("Volume > 1.2× 20-bar average")
        if _finite(macd) and _finite(macdsig) and macd > macdsig:
            conf += 0.05
            rationale.append("MACD above signal line")
        risk = entry - stop
        reward = target - entry
        if risk > 0 and reward > 0:
            setups.append(
                TradeSetup(
                    name="Range breakout",
                    direction="long",
                    entry=entry,
                    stop=float(stop),
                    target=float(target),
                    risk=float(risk),
                    reward=float(reward),
                    rr=float(reward / risk),
                    confidence=min(conf, 0.90),
                    score=0.0,
                    rationale=[regime.rationale, *rationale],
                    tags=["breakout", "long"],
                )
            )

    # ── 6. Range breakout SHORT ────────────────────────────────────
    if "breakout" in regime.eligible_tags and _finite(atr) and _finite(low20) and close < low20:
        entry = close
        stop = entry + 1.5 * atr
        range_size = high20 - low20
        target = entry - max(2.0 * atr, range_size)
        rationale = [
            f"Close {close:.2f} < prior 20-bar low {low20:.2f}",
            f"Project range {range_size:.2f} vs 2×ATR {2*atr:.2f}",
        ]
        conf = 0.40
        if _finite(adx) and adx >= 18:
            conf += 0.10
            rationale.append(f"ADX {adx:.1f} confirms momentum")
        if len(v) >= 21 and v[-21:-1].mean() > 0 and v[-1] > v[-21:-1].mean() * 1.2:
            conf += 0.10
            rationale.append("Volume > 1.2× 20-bar average")
        if _finite(macd) and _finite(macdsig) and macd < macdsig:
            conf += 0.05
            rationale.append("MACD below signal line")
        risk = stop - entry
        reward = entry - target
        if risk > 0 and reward > 0:
            setups.append(
                TradeSetup(
                    name="Range breakout",
                    direction="short",
                    entry=entry,
                    stop=float(stop),
                    target=float(target),
                    risk=float(risk),
                    reward=float(reward),
                    rr=float(reward / risk),
                    confidence=min(conf, 0.90),
                    score=0.0,
                    rationale=[regime.rationale, *rationale],
                    tags=["breakout", "short"],
                )
            )

    # Rank: rr × confidence, drop anything below RR=1.
    for s in setups:
        s.score = s.rr * s.confidence
    setups = [s for s in setups if s.rr >= 1.0]
    setups.sort(key=lambda s: s.score, reverse=True)
    return [asdict(s) for s in setups[:max_setups]]
