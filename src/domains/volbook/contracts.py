"""Futures contract helpers for the volatility book.

Canonical home for the small set of futures the desk watches. Each
symbol carries its exchange and a label template so the CLI can accept
short aliases (``CL_JUN26``) and the canvas can render a human-readable
contract name.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

# IB month codes used by ``lastTradeDateOrContractMonth`` when expressed
# as ``YYYYMM``. The mapping is only used for pretty labels.
_MONTH_LABELS: Mapping[str, str] = {
    "01": "Jan", "02": "Feb", "03": "Mar", "04": "Apr",
    "05": "May", "06": "Jun", "07": "Jul", "08": "Aug",
    "09": "Sep", "10": "Oct", "11": "Nov", "12": "Dec",
}


@dataclass(frozen=True)
class FuturesSpec:
    """Everything IB needs to qualify a futures contract."""

    symbol: str
    exchange: str
    expiry: str          # IB lastTradeDateOrContractMonth, e.g. "202606"
    currency: str = "USD"
    display_symbol: str | None = None
    multiplier: str | None = None
    trading_class: str | None = None

    @property
    def label_symbol(self) -> str:
        return self.display_symbol or self.symbol

    @property
    def label(self) -> str:
        """Human-readable label, e.g. ``CL Jun'26``."""
        if len(self.expiry) >= 6:
            yyyy, mm = self.expiry[:4], self.expiry[4:6]
            month = _MONTH_LABELS.get(mm, mm)
            yy = yyyy[-2:]
            return f"{self.label_symbol} {month}'{yy}"
        return f"{self.label_symbol} {self.expiry}"

    @property
    def key(self) -> str:
        """Stable identifier used in the bundle."""
        return f"{self.label_symbol}-{self.expiry}"

    def with_expiry(self, expiry: str) -> "FuturesSpec":
        """Return the same contract root pointed at another expiry."""
        return FuturesSpec(
            symbol=self.symbol,
            exchange=self.exchange,
            expiry=expiry,
            currency=self.currency,
            display_symbol=self.display_symbol,
            multiplier=self.multiplier,
            trading_class=self.trading_class,
        )


# Curated registry of the desk's usual suspects. Aliases are uppercased
# so the CLI ``--alias`` path is case-insensitive.
KNOWN_FUTURES: Mapping[str, FuturesSpec] = {
    "SR3_JUN26": FuturesSpec(symbol="SR3", exchange="CME", expiry="202606"),
    "CL_JUN26": FuturesSpec(symbol="CL", exchange="NYMEX", expiry="202606"),
    "CL_JUL26": FuturesSpec(symbol="CL", exchange="NYMEX", expiry="202607"),
    "CL_AUG26": FuturesSpec(symbol="CL", exchange="NYMEX", expiry="202608"),
    "NG_JUN26": FuturesSpec(symbol="NG", exchange="NYMEX", expiry="202606"),
    "RB_JUN26": FuturesSpec(symbol="RB", exchange="NYMEX", expiry="202606"),
    "HO_JUN26": FuturesSpec(symbol="HO", exchange="NYMEX", expiry="202606"),
    "GC_JUN26": FuturesSpec(symbol="GC", exchange="COMEX", expiry="202606"),
    "SI_JUN26": FuturesSpec(
        symbol="SI",
        exchange="COMEX",
        expiry="202606",
        multiplier="5000",
        trading_class="SI",
    ),
    "HG_JUN26": FuturesSpec(symbol="HG", exchange="COMEX", expiry="202606"),
    "ES_JUN26": FuturesSpec(symbol="ES", exchange="CME", expiry="202606"),
    "NQ_JUN26": FuturesSpec(symbol="NQ", exchange="CME", expiry="202606"),
    "RTY_JUN26": FuturesSpec(symbol="RTY", exchange="CME", expiry="202606"),
    "SDA_JUN26": FuturesSpec(symbol="SDA", exchange="CME", expiry="202606"),
    "SME_JUN26": FuturesSpec(symbol="SME", exchange="CME", expiry="202606"),
    "ZN_JUN26": FuturesSpec(symbol="ZN", exchange="CBOT", expiry="202606"),
    "ZF_JUN26": FuturesSpec(symbol="ZF", exchange="CBOT", expiry="202606"),
    "ZT_JUN26": FuturesSpec(symbol="ZT", exchange="CBOT", expiry="202606"),
    "ZB_JUN26": FuturesSpec(symbol="ZB", exchange="CBOT", expiry="202606"),
    "TN_JUN26": FuturesSpec(symbol="TN", exchange="CBOT", expiry="202606"),
    "UB_JUN26": FuturesSpec(symbol="UB", exchange="CBOT", expiry="202606"),
    "ZC_JUN26": FuturesSpec(symbol="ZC", exchange="CBOT", expiry="202606"),
    "ZS_JUN26": FuturesSpec(symbol="ZS", exchange="CBOT", expiry="202606"),
    "ZL_JUN26": FuturesSpec(symbol="ZL", exchange="CBOT", expiry="202606"),
    "ZM_JUN26": FuturesSpec(symbol="ZM", exchange="CBOT", expiry="202606"),
    "ZW_JUN26": FuturesSpec(symbol="ZW", exchange="CBOT", expiry="202606"),
    "KE_JUN26": FuturesSpec(symbol="KE", exchange="CBOT", expiry="202606"),
    "LE_JUN26": FuturesSpec(symbol="LE", exchange="CME", expiry="202606"),
    "HE_JUN26": FuturesSpec(symbol="HE", exchange="CME", expiry="202606"),
    "GF_JUN26": FuturesSpec(symbol="GF", exchange="CME", expiry="202606"),
    "DC_JUN26": FuturesSpec(symbol="DC", exchange="CME", expiry="202606"),
    "GDK_JUN26": FuturesSpec(symbol="GDK", exchange="CME", expiry="202606"),
    "CSC_JUN26": FuturesSpec(symbol="CSC", exchange="CME", expiry="202606"),
    "CB_JUN26": FuturesSpec(symbol="CB", exchange="CME", expiry="202606"),
    "GNF_JUN26": FuturesSpec(symbol="GNF", exchange="CME", expiry="202606"),
    "DY_JUN26": FuturesSpec(symbol="DY", exchange="CME", expiry="202606"),
    "PL_JUN26": FuturesSpec(symbol="PL", exchange="NYMEX", expiry="202606"),
    "PA_JUN26": FuturesSpec(symbol="PA", exchange="NYMEX", expiry="202606"),
    "MET_JUN26": FuturesSpec(symbol="MET", exchange="CME", expiry="202606"),
    "MBT_JUN26": FuturesSpec(symbol="MBT", exchange="CME", expiry="202606"),
    "6E_JUN26": FuturesSpec(
        symbol="EUR",
        exchange="CME",
        expiry="202606",
        display_symbol="6E",
        multiplier="125000",
        trading_class="6E",
    ),
    "6J_JUN26": FuturesSpec(
        symbol="JPY",
        exchange="CME",
        expiry="202606",
        display_symbol="6J",
        multiplier="12500000",
        trading_class="6J",
    ),
    "6A_JUN26": FuturesSpec(
        symbol="AUD",
        exchange="CME",
        expiry="202606",
        display_symbol="6A",
        multiplier="100000",
        trading_class="6A",
    ),
    "6B_JUN26": FuturesSpec(
        symbol="GBP",
        exchange="CME",
        expiry="202606",
        display_symbol="6B",
        multiplier="62500",
        trading_class="6B",
    ),
    "6C_JUN26": FuturesSpec(
        symbol="CAD",
        exchange="CME",
        expiry="202606",
        display_symbol="6C",
        multiplier="100000",
        trading_class="6C",
    ),
    "6S_JUN26": FuturesSpec(
        symbol="CHF",
        exchange="CME",
        expiry="202606",
        display_symbol="6S",
        multiplier="125000",
        trading_class="6S",
    ),
    "VX_JUN26": FuturesSpec(
        symbol="VIX",
        exchange="CFE",
        expiry="202606",
        display_symbol="VX",
        multiplier="1000",
        trading_class="VX",
    ),
}

CORE_MACRO_ALIASES: tuple[str, ...] = (
    "CL_JUN26",
    "NG_JUN26",
    "RB_JUN26",
    "HO_JUN26",
    "GC_JUN26",
    "SI_JUN26",
    "HG_JUN26",
    "ES_JUN26",
    "NQ_JUN26",
    "RTY_JUN26",
    "ZN_JUN26",
    "ZB_JUN26",
    "6E_JUN26",
    "6J_JUN26",
)

OPTIONS_UNDERLYING_ALIASES: tuple[str, ...] = (
    "SR3_JUN26",
    "ES_JUN26",
    "ZN_JUN26",
    "CL_JUN26",
    "NG_JUN26",
    "ZF_JUN26",
    "ZC_JUN26",
    "ZB_JUN26",
    "ZT_JUN26",
    "ZS_JUN26",
    "GC_JUN26",
    "ZL_JUN26",
    "SDA_JUN26",
    "6E_JUN26",
    "LE_JUN26",
    "ZW_JUN26",
    "HE_JUN26",
    "ZM_JUN26",
    "NQ_JUN26",
    "KE_JUN26",
    "6J_JUN26",
    "GF_JUN26",
    "SME_JUN26",
    "SI_JUN26",
    "DC_JUN26",
    "HG_JUN26",
    "6A_JUN26",
    "6B_JUN26",
    "RTY_JUN26",
    "HO_JUN26",
    "6C_JUN26",
    "GDK_JUN26",
    "CSC_JUN26",
    "MET_JUN26",
    "GNF_JUN26",
    "CB_JUN26",
    "6S_JUN26",
    "PL_JUN26",
    "MBT_JUN26",
    "RB_JUN26",
    "DY_JUN26",
    "PA_JUN26",
)

# Listed contract-month patterns per root, keyed by IB ``symbol``. Months
# are 1-indexed ints. Patterns capture only the *frequently listed*
# expiries — IB occasionally adds serial months beyond the cycle, but
# those have negligible volume / data depth so the dated walker can
# safely skip them.
#
# Source: each exchange's contract specs page (CME, CBOT, NYMEX,
# COMEX, ICE). Cross-checked against IBKR's product page.
DEFAULT_MONTH_PATTERN: tuple[int, ...] = tuple(range(1, 13))
_HMUZ = (3, 6, 9, 12)
_HKNUZ = (3, 5, 7, 9, 12)
_GJMQVZ = (2, 4, 6, 8, 10, 12)
_FHJKQUVX = (1, 3, 4, 5, 8, 9, 10, 11)
MONTH_PATTERN_BY_ROOT: Mapping[str, tuple[int, ...]] = {
    # Equities (quarterly)
    "ES": _HMUZ,
    "NQ": _HMUZ,
    "RTY": _HMUZ,
    "MES": _HMUZ,
    "MNQ": _HMUZ,
    # Rates (quarterly)
    "ZN": _HMUZ,
    "ZB": _HMUZ,
    "ZF": _HMUZ,
    "ZT": _HMUZ,
    "TN": _HMUZ,
    "UB": _HMUZ,
    "SR3": _HMUZ,
    # FX (quarterly)
    "6A": _HMUZ,
    "6B": _HMUZ,
    "6C": _HMUZ,
    "6E": _HMUZ,
    "6J": _HMUZ,
    "6S": _HMUZ,
    # Volatility (Cboe VIX futures list every calendar month)
    "VIX": DEFAULT_MONTH_PATTERN,
    # Agriculture (5-cycle HKNUZ)
    "ZC": _HKNUZ,
    "ZS": _HKNUZ,
    "ZW": _HKNUZ,
    "ZM": _HKNUZ,
    "ZL": _HKNUZ,
    "KE": _HKNUZ,
    # Metals (bi-monthly)
    "GC": _GJMQVZ,
    "SI": _GJMQVZ,
    "HG": _GJMQVZ,
    "PL": _GJMQVZ,
    "PA": _GJMQVZ,
    # Livestock
    "LE": _GJMQVZ,
    "GF": _FHJKQUVX,
    # Energy & lean hogs (every month)
    "CL": DEFAULT_MONTH_PATTERN,
    "NG": DEFAULT_MONTH_PATTERN,
    "HO": DEFAULT_MONTH_PATTERN,
    "RB": DEFAULT_MONTH_PATTERN,
    "HE": DEFAULT_MONTH_PATTERN,
    # Crypto / micro / weather / dairy / event — default to every month
    # so the qualifier filters real listings naturally.
    "MBT": DEFAULT_MONTH_PATTERN,
    "MET": DEFAULT_MONTH_PATTERN,
    "DC": DEFAULT_MONTH_PATTERN,
    "DY": DEFAULT_MONTH_PATTERN,
    "GDK": DEFAULT_MONTH_PATTERN,
    "CB": DEFAULT_MONTH_PATTERN,
    "CSC": DEFAULT_MONTH_PATTERN,
    "GNF": DEFAULT_MONTH_PATTERN,
    "SDA": DEFAULT_MONTH_PATTERN,
    "SME": DEFAULT_MONTH_PATTERN,
}


def month_pattern_for_root(symbol: str) -> tuple[int, ...]:
    """Return the cycle of listed months for ``symbol`` (1-indexed)."""
    return MONTH_PATTERN_BY_ROOT.get(symbol, DEFAULT_MONTH_PATTERN)


# Roots that IBKR does not expose as a qualifiable ``ContFuture`` (the
# continuous historical request returns "Error 200: No security definition
# has been found"). The continuous-minute refresh skips these so a routine
# run can still exit 0; their deep history continues to flow through the
# dated walk (``refresh_dated_futures_minute`` / ``walk_dated_futures_minute``),
# which qualifies explicit dated contracts instead of a ContFuture.
CONTINUOUS_INELIGIBLE_ROOTS: frozenset[str] = frozenset({
    "SR3",  # 3M SOFR - no ContFuture security definition on IBKR
    "SDA",  # no ContFuture security definition
    "SME",  # no ContFuture security definition
    "DC",   # Class III Milk - no ContFuture security definition
    "GNF",  # Nonfat Dry Milk - no ContFuture security definition
})


def is_continuous_eligible(symbol: str) -> bool:
    """True if ``symbol`` can be requested as an IBKR ``ContFuture``."""
    return symbol.upper() not in CONTINUOUS_INELIGIBLE_ROOTS


def enumerate_dated_specs(
    spec: FuturesSpec,
    *,
    min_expiry: str,
    max_expiry: str,
) -> list[FuturesSpec]:
    """Generate dated ``FuturesSpec``s using the root's month pattern.

    No IB calls — purely client-side. The walker uses this in lieu of
    ``reqContractDetails(includeExpired=True)``, which is unreliable
    (queues silently in TWS for popular roots like ES). Months outside
    the root's listed pattern are still included if the pattern entry
    is :data:`DEFAULT_MONTH_PATTERN`, so seldom-traded products fall
    back to "try every month and let the qualifier filter".
    """
    if len(min_expiry) < 6 or len(max_expiry) < 6 or min_expiry > max_expiry:
        return []
    months = month_pattern_for_root(spec.symbol)
    start_year = int(min_expiry[:4])
    start_month = int(min_expiry[4:6])
    end_year = int(max_expiry[:4])
    end_month = int(max_expiry[4:6])
    out: list[FuturesSpec] = []
    for year in range(start_year, end_year + 1):
        for m in months:
            if year == start_year and m < start_month:
                continue
            if year == end_year and m > end_month:
                continue
            out.append(spec.with_expiry(f"{year:04d}{m:02d}"))
    return out


ASSET_CLASS_BY_ROOT: Mapping[str, str] = {
    "SR3": "Interest Rates",
    "ZN": "Interest Rates",
    "ZF": "Interest Rates",
    "ZT": "Interest Rates",
    "ZB": "Interest Rates",
    "TN": "Interest Rates",
    "UB": "Interest Rates",
    "ES": "Equities",
    "NQ": "Equities",
    "RTY": "Equities",
    "SDA": "Equities",
    "SME": "Equities",
    "CL": "Energy",
    "NG": "Energy",
    "RB": "Energy",
    "HO": "Energy",
    "GC": "Metals",
    "SI": "Metals",
    "HG": "Metals",
    "PL": "Metals",
    "PA": "Metals",
    "ZC": "Agriculture",
    "ZS": "Agriculture",
    "ZL": "Agriculture",
    "ZM": "Agriculture",
    "ZW": "Agriculture",
    "KE": "Agriculture",
    "LE": "Agriculture",
    "HE": "Agriculture",
    "GF": "Agriculture",
    "DC": "Agriculture",
    "GDK": "Agriculture",
    "CSC": "Agriculture",
    "CB": "Agriculture",
    "GNF": "Agriculture",
    "DY": "Agriculture",
    "6E": "FX",
    "6J": "FX",
    "6A": "FX",
    "6B": "FX",
    "6C": "FX",
    "6S": "FX",
    "MET": "Crypto",
    "MBT": "Crypto",
    "VX": "Volatility",
}

CATEGORY_ORDER: tuple[str, ...] = (
    "Interest Rates",
    "Equities",
    "Energy",
    "Metals",
    "Agriculture",
    "FX",
    "Crypto",
    "Volatility",
)

# Exchange picked when the caller supplies only ``--symbol``. Keeps the
# common path (``--symbol CL``) one-argument for the desk.
_DEFAULT_EXCHANGE: Mapping[str, str] = {
    "SR3": "CME",
    "CL": "NYMEX",
    "NG": "NYMEX",
    "HO": "NYMEX",
    "RB": "NYMEX",
    "GC": "COMEX",
    "SI": "COMEX",
    "HG": "COMEX",
    "ES": "CME",
    "NQ": "CME",
    "RTY": "CME",
    "SDA": "CME",
    "SME": "CME",
    "ZN": "CBOT",
    "ZF": "CBOT",
    "ZT": "CBOT",
    "ZB": "CBOT",
    "TN": "CBOT",
    "UB": "CBOT",
    "ZC": "CBOT",
    "ZS": "CBOT",
    "ZL": "CBOT",
    "ZM": "CBOT",
    "ZW": "CBOT",
    "KE": "CBOT",
    "LE": "CME",
    "HE": "CME",
    "GF": "CME",
    "DC": "CME",
    "GDK": "CME",
    "CSC": "CME",
    "CB": "CME",
    "GNF": "CME",
    "DY": "CME",
    "PL": "NYMEX",
    "PA": "NYMEX",
    "MET": "CME",
    "MBT": "CME",
    "6E": "CME",
    "6J": "CME",
    "6A": "CME",
    "6B": "CME",
    "6C": "CME",
    "6S": "CME",
    "VIX": "CFE",
}

_SYMBOL_OVERRIDES: Mapping[str, FuturesSpec] = {
    "VX": FuturesSpec(
        symbol="VIX",
        exchange="CFE",
        expiry="",
        display_symbol="VX",
        multiplier="1000",
        trading_class="VX",
    ),
    "6E": FuturesSpec(
        symbol="EUR",
        exchange="CME",
        expiry="",
        display_symbol="6E",
        multiplier="125000",
        trading_class="6E",
    ),
    "6J": FuturesSpec(
        symbol="JPY",
        exchange="CME",
        expiry="",
        display_symbol="6J",
        multiplier="12500000",
        trading_class="6J",
    ),
    "6A": FuturesSpec(
        symbol="AUD",
        exchange="CME",
        expiry="",
        display_symbol="6A",
        multiplier="100000",
        trading_class="6A",
    ),
    "6B": FuturesSpec(
        symbol="GBP",
        exchange="CME",
        expiry="",
        display_symbol="6B",
        multiplier="62500",
        trading_class="6B",
    ),
    "6C": FuturesSpec(
        symbol="CAD",
        exchange="CME",
        expiry="",
        display_symbol="6C",
        multiplier="100000",
        trading_class="6C",
    ),
    "6S": FuturesSpec(
        symbol="CHF",
        exchange="CME",
        expiry="",
        display_symbol="6S",
        multiplier="125000",
        trading_class="6S",
    ),
}


def resolve_futures_spec(
    alias: str | None = None,
    *,
    symbol: str | None = None,
    expiry: str | None = None,
    exchange: str | None = None,
    currency: str = "USD",
) -> FuturesSpec:
    """Resolve CLI flags / alias into a concrete ``FuturesSpec``.

    Precedence:
      1. ``alias`` — look up in ``KNOWN_FUTURES``.
      2. ``symbol + expiry`` with exchange defaulted from ``_DEFAULT_EXCHANGE``.
    """
    if alias:
        key = alias.upper()
        if key not in KNOWN_FUTURES:
            raise KeyError(
                f"unknown futures alias {alias!r}; known: {sorted(KNOWN_FUTURES)}"
            )
        return KNOWN_FUTURES[key]

    if not symbol or not expiry:
        raise ValueError("must supply either --alias or both --symbol and --expiry")

    sym = symbol.upper()
    if sym in _SYMBOL_OVERRIDES:
        base = _SYMBOL_OVERRIDES[sym]
        return FuturesSpec(
            symbol=base.symbol,
            exchange=exchange or base.exchange,
            expiry=expiry,
            currency=currency,
            display_symbol=base.display_symbol,
            multiplier=base.multiplier,
            trading_class=base.trading_class,
        )

    ex = exchange or _DEFAULT_EXCHANGE.get(sym)
    if not ex:
        raise ValueError(
            f"no default exchange for symbol {symbol!r}; pass --exchange explicitly"
        )
    return FuturesSpec(symbol=sym, exchange=ex, expiry=expiry, currency=currency)
