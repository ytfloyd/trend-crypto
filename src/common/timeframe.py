from __future__ import annotations

import re


def timeframe_to_seconds(tf: str) -> int:
    tf = tf.lower().strip()
    m = re.match(r"(\d+)([hdm])", tf)
    if not m:
        raise ValueError(f"Unsupported timeframe: {tf}")
    value = int(m.group(1))
    unit = m.group(2)
    if unit == "m":
        return value * 60
    if unit == "h":
        return value * 3600
    if unit == "d":
        return value * 24 * 3600
    raise ValueError(f"Unsupported timeframe: {tf}")


def periods_per_year_from_timeframe(tf: str) -> float:
    sec = timeframe_to_seconds(tf)
    return (365 * 24 * 3600) / sec


def hours_per_bar(tf: str) -> float:
    return timeframe_to_seconds(tf) / 3600

