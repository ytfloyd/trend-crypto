"""Handcrafted weight allocation following Carver's top-down grouping.

Instead of optimising weights (which overfits in small samples), Carver
advocates a principled "handcrafting" approach:

1. Group similar items (rules or instruments) into families.
2. Equal weight within each group.
3. Equal weight across groups (or slight tilts based on prior beliefs).

This produces robust, diversified allocations without in-sample fitting.

Reference: Robert Carver, *Systematic Trading*, Chapter 9.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class HandcraftedWeights:
    """Top-down weight allocator using Carver's grouping heuristic.

    Parameters
    ----------
    groups : dict mapping group name to a list of item names.
        E.g. ``{"trend_ma": ["ewmac_8_32", "ewmac_16_64"], "breakout": ["bo_20", "bo_40"]}``
    group_weights : optional dict mapping group name to a relative weight.
        If None, groups are equally weighted.
    """

    groups: dict[str, list[str]]
    group_weights: dict[str, float] | None = None
    _weights: dict[str, float] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self._compute()

    def _compute(self) -> None:
        if self.group_weights is not None:
            gw = self.group_weights
        else:
            gw = {g: 1.0 for g in self.groups}

        total_gw = sum(gw.values())
        if total_gw <= 0:
            raise ValueError("Group weights must sum to a positive number")
        norm_gw = {g: w / total_gw for g, w in gw.items()}

        weights: dict[str, float] = {}
        for group_name, members in self.groups.items():
            group_alloc = norm_gw.get(group_name, 0.0)
            if not members:
                continue
            per_member = group_alloc / len(members)
            for m in members:
                weights[m] = weights.get(m, 0.0) + per_member

        self._weights = weights

    @property
    def weights(self) -> dict[str, float]:
        """Final weights for all items, normalised to sum to 1.0."""
        return dict(self._weights)

    def get(self, name: str, default: float = 0.0) -> float:
        return self._weights.get(name, default)

    def as_list(self) -> list[tuple[str, float]]:
        """Sorted (name, weight) pairs."""
        return sorted(self._weights.items(), key=lambda x: -x[1])

    def summary(self) -> str:
        """Human-readable summary of the allocation."""
        lines = ["Handcrafted Weight Allocation", "=" * 40]

        group_totals: dict[str, float] = {}
        for group_name, members in self.groups.items():
            grp_total = sum(self._weights.get(m, 0.0) for m in members)
            group_totals[group_name] = grp_total

        for group_name, members in self.groups.items():
            lines.append(f"\n  {group_name} ({group_totals[group_name]:.1%} total):")
            for m in members:
                lines.append(f"    {m:30s} {self._weights.get(m, 0.0):.4f}")

        lines.append(f"\n  Total: {sum(self._weights.values()):.4f}")
        return "\n".join(lines)


# ── Convenience constructors ─────────────────────────────────────────────

def handcraft_forecast_weights(
    trend_rules: list[str] | None = None,
    breakout_rules: list[str] | None = None,
    momentum_rules: list[str] | None = None,
    carry_rules: list[str] | None = None,
    group_weights: dict[str, float] | None = None,
) -> HandcraftedWeights:
    """Build handcrafted forecast weights from rule families.

    Typical usage for a trend-following system:

        >>> hw = handcraft_forecast_weights(
        ...     trend_rules=["ewmac_8_32", "ewmac_16_64", "ewmac_32_128"],
        ...     breakout_rules=["breakout_20", "breakout_40"],
        ... )
        >>> hw.weights
        {'ewmac_8_32': 0.1667, 'ewmac_16_64': 0.1667, ...}
    """
    groups: dict[str, list[str]] = {}
    if trend_rules:
        groups["trend"] = trend_rules
    if breakout_rules:
        groups["breakout"] = breakout_rules
    if momentum_rules:
        groups["momentum"] = momentum_rules
    if carry_rules:
        groups["carry"] = carry_rules

    if not groups:
        raise ValueError("At least one rule family must be provided")

    return HandcraftedWeights(groups=groups, group_weights=group_weights)


def handcraft_instrument_weights(
    groups: dict[str, list[str]],
    group_weights: dict[str, float] | None = None,
) -> HandcraftedWeights:
    """Build handcrafted instrument weights from asset classes.

    Example:

        >>> hw = handcraft_instrument_weights({
        ...     "large_cap": ["BTC-USD", "ETH-USD"],
        ...     "mid_cap":   ["SOL-USD", "AVAX-USD", "LINK-USD"],
        ... })
        >>> hw.weights
        {'BTC-USD': 0.25, 'ETH-USD': 0.25, 'SOL-USD': 0.1667, ...}
    """
    return HandcraftedWeights(groups=groups, group_weights=group_weights)
