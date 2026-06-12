"""Databento loader for deep CL (NYMEX WTI) 1-minute history.

Fetches dated CL contracts from Databento's ``GLBX.MDP3`` dataset (``ohlcv-1m``
schema) and writes them into the same volbook minute lake the IBKR walker uses,
under ``expiry='YYYYMM'``.  The institutional continuous is then rebuilt by the
existing ``construct_continuous_series`` roll engine -- this module adds *no* new
roll logic, it only supplies more dated contracts than IBKR retains (IBKR purges
expired CL near-months after ~2 years; Databento goes back to 2010-06).

Cost discipline
---------------
Databento meters historical usage by uncompressed binary size ($/GB) and exposes
a free ``metadata.get_cost`` endpoint.  :meth:`DatabentoLoader.estimate_cost`
sums per-contract costs over only each contract's *front + roll* window
(``window_days`` before last-trade) so a quote can be produced -- and the spend
minimised -- before any paid download.  No paid call happens unless
:meth:`fetch_bars` / :meth:`backfill_contract` is invoked explicitly.
"""
from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, List, Sequence

from .bundle import Bar
from .continuous import WeekendHolidayCalendar, cl_last_trade_date

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

    from .datalake import MinuteLake

DATASET = "GLBX.MDP3"
SCHEMA = "ohlcv-1m"
STYPE_IN = "raw_symbol"
# Billing mode for the historical-streaming path (``timeseries.get_range``).
# Passing it explicitly to ``get_cost`` makes the quote match what a download
# actually bills.
MODE = "historical-streaming"

# CME month codes, index 0 -> January ... index 11 -> December.
_CME_MONTH_CODES = "FGHJKMNQUVXZ"


@dataclass(frozen=True)
class ContractWindow:
    """A single dated-contract request window for the deep backfill."""

    expiry: str          # YYYYMM (delivery month)
    raw_symbol: str      # Databento/CME native symbol, e.g. "CLF0"
    start: datetime      # UTC, inclusive
    end: datetime        # UTC, exclusive


def cl_raw_symbol(expiry: str) -> str:
    """Map a ``YYYYMM`` delivery month to its CME/Databento raw symbol.

    CME Globex uses a single-digit year (e.g. Jan-2020 -> ``CLF0``).  The digit
    repeats every decade, but each per-contract request below is scoped to a
    tight window around that contract's last-trade date, so the symbol is
    unambiguous within the requested range.
    """
    if len(expiry) < 6:
        raise ValueError(f"expiry must be YYYYMM, got {expiry!r}")
    year = int(expiry[:4])
    month = int(expiry[4:6])
    if not 1 <= month <= 12:
        raise ValueError(f"invalid month in expiry {expiry!r}")
    return f"CL{_CME_MONTH_CODES[month - 1]}{year % 10}"


def contract_window(expiry: str, *, window_days: int = 45) -> tuple[datetime, datetime]:
    """Return the ``(start, end)`` UTC window for a CL contract.

    The window spans ``window_days`` before the contract's last-trade date
    through the day after last-trade.  That covers the period the contract is
    the front month (plus the roll overlap with its successor) -- sufficient for
    the volume-crossover roll engine and far cheaper than each contract's full
    multi-year life.
    """
    year = int(expiry[:4])
    month = int(expiry[4:6])
    ltd = cl_last_trade_date(date(year, month, 1), calendar=WeekendHolidayCalendar())
    start = datetime(ltd.year, ltd.month, ltd.day, tzinfo=timezone.utc) - timedelta(days=window_days)
    end = datetime(ltd.year, ltd.month, ltd.day, tzinfo=timezone.utc) + timedelta(days=1)
    return start, end


def enumerate_cl_expiries(start_expiry: str, end_expiry: str) -> List[str]:
    """All monthly CL ``YYYYMM`` delivery months in ``[start, end]`` inclusive."""
    if len(start_expiry) < 6 or len(end_expiry) < 6 or start_expiry > end_expiry:
        return []
    y, m = int(start_expiry[:4]), int(start_expiry[4:6])
    ey, em = int(end_expiry[:4]), int(end_expiry[4:6])
    out: List[str] = []
    while (y, m) <= (ey, em):
        out.append(f"{y:04d}{m:02d}")
        m += 1
        if m > 12:
            m = 1
            y += 1
    return out


def _df_to_bars(df: "pd.DataFrame") -> List[Bar]:
    """Convert a Databento ``ohlcv-1m`` DataFrame to ``Bar`` rows.

    Expects a tz-aware (UTC) ``ts_event`` index and float ``open/high/low/
    close/volume`` columns (Databento ``DBNStore.to_df(price_type='float')``).
    """
    bars: List[Bar] = []
    for ts, row in df.iterrows():
        py_ts = ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts
        if py_ts.tzinfo is None:
            py_ts = py_ts.replace(tzinfo=timezone.utc)
        else:
            py_ts = py_ts.astimezone(timezone.utc)
        bars.append(
            Bar(
                t=py_ts.isoformat(),
                o=float(row["open"]),
                h=float(row["high"]),
                l=float(row["low"]),
                c=float(row["close"]),
                v=float(row["volume"]),
            )
        )
    return bars


class DatabentoLoader:
    """Thin wrapper over ``databento.Historical`` for the CL deep backfill."""

    def __init__(self, api_key: str | None = None, *, client: Any | None = None) -> None:
        if client is not None:
            self._client = client
            return
        key = api_key or os.environ.get("DATABENTO_API_KEY")
        if not key:
            raise RuntimeError(
                "DATABENTO_API_KEY not set; export it or pass api_key=/client=."
            )
        import databento as db  # imported lazily so the module loads without the dep

        self._client = db.Historical(key)

    def windows(
        self, start_expiry: str, end_expiry: str, *, window_days: int = 45
    ) -> List[ContractWindow]:
        """Build the per-contract request windows for the backfill range."""
        out: List[ContractWindow] = []
        for expiry in enumerate_cl_expiries(start_expiry, end_expiry):
            start, end = contract_window(expiry, window_days=window_days)
            out.append(
                ContractWindow(
                    expiry=expiry,
                    raw_symbol=cl_raw_symbol(expiry),
                    start=start,
                    end=end,
                )
            )
        return out

    def dataset_range(self) -> tuple[str, str]:
        """Return the dataset's available ``(start, end)`` (free metadata call)."""
        rng = self._client.metadata.get_dataset_range(dataset=DATASET)
        return str(rng["start"]), str(rng["end"])

    def unit_price_usd_per_gb(self) -> float:
        """Live ohlcv-1m historical-streaming unit price ($/GB) for the dataset.

        Falls back to the published 280.0 if the price list cannot be parsed.
        Free metadata call.
        """
        try:
            prices = self._client.metadata.list_unit_prices(dataset=DATASET)
        except Exception:  # noqa: BLE001 - pricing is best-effort, never blocks
            return 280.0
        for entry in prices or []:
            if not isinstance(entry, dict):
                continue
            mode = str(entry.get("mode", "")).lower()
            if MODE not in mode and "stream" not in mode:
                continue
            schemas = entry.get("unit_prices") or entry.get("prices") or entry
            if isinstance(schemas, dict):
                for key, val in schemas.items():
                    if str(key).lower() == SCHEMA:
                        try:
                            return float(val)
                        except (TypeError, ValueError):
                            pass
        return 280.0

    def _clamp_start(self, clamp_to_dataset: bool) -> datetime | None:
        if not clamp_to_dataset:
            return None
        ds_start, _ = self.dataset_range()
        return _parse_utc(ds_start)

    def contract_cost(
        self,
        window: ContractWindow,
        *,
        clamp_start: datetime | None = None,
    ) -> tuple[float, datetime | None]:
        """Return ``(cost_usd, effective_start)`` for one contract window.

        Cost is ``0.0`` with ``effective_start=None`` if the (clamped) window is
        empty (entirely before the dataset start). Free metadata call.
        """
        req_start = window.start
        if clamp_start is not None and req_start < clamp_start:
            req_start = clamp_start
        if req_start >= window.end:
            return 0.0, None
        cost = float(self._get_cost(window.raw_symbol, req_start, window.end))
        return cost, req_start

    def _get_cost(self, raw_symbol: str, start: datetime, end: datetime) -> float:
        """Call ``metadata.get_cost`` for the streaming mode (free, exact).

        ``mode`` is deprecated in newer SDKs (historical billing is now uniform);
        we pass it for explicitness where supported and fall back if removed,
        suppressing the deprecation warning either way.
        """
        kwargs = dict(
            dataset=DATASET,
            symbols=[raw_symbol],
            schema=SCHEMA,
            stype_in=STYPE_IN,
            start=start.isoformat(),
            end=end.isoformat(),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            try:
                return self._client.metadata.get_cost(mode=MODE, **kwargs)
            except TypeError:
                return self._client.metadata.get_cost(**kwargs)

    def estimate_cost(
        self,
        start_expiry: str,
        end_expiry: str,
        *,
        window_days: int = 45,
        clamp_to_dataset: bool = True,
    ) -> dict[str, Any]:
        """Sum ``metadata.get_cost`` over each contract window (no data pulled).

        Returns a dict with the total USD cost and a per-contract breakdown.
        ``metadata.get_cost`` is free; this never triggers a billable download.
        """
        windows = self.windows(start_expiry, end_expiry, window_days=window_days)
        clamp_start = self._clamp_start(clamp_to_dataset)

        rows: List[dict[str, Any]] = []
        total = 0.0
        for w in windows:
            cost, eff_start = self.contract_cost(w, clamp_start=clamp_start)
            if eff_start is None:
                rows.append(
                    {
                        "expiry": w.expiry,
                        "raw_symbol": w.raw_symbol,
                        "skipped": "before_dataset_start",
                        "cost_usd": 0.0,
                    }
                )
                continue
            total += cost
            rows.append(
                {
                    "expiry": w.expiry,
                    "raw_symbol": w.raw_symbol,
                    "start": eff_start.isoformat(),
                    "end": w.end.isoformat(),
                    "cost_usd": cost,
                }
            )
        return {
            "dataset": DATASET,
            "schema": SCHEMA,
            "rate_usd_per_gb": self.unit_price_usd_per_gb(),
            "contracts": len(windows),
            "total_cost_usd": total,
            "breakdown": rows,
        }

    def plan_within_budget(
        self,
        start_expiry: str,
        end_expiry: str,
        *,
        cap_usd: float,
        window_days: int = 45,
        order: str = "newest",
        exclude_expiries: Sequence[str] = (),
        clamp_to_dataset: bool = True,
    ) -> dict[str, Any]:
        """Select the largest prefix of contracts whose total cost fits ``cap_usd``.

        Contracts in ``exclude_expiries`` (e.g. already present in the lake) are
        dropped before costing so a resumed run never re-pays for stored data.
        Ordering is ``newest`` (default) or ``oldest`` by expiry. Pure metadata;
        no billable download. The returned ``selected`` list is guaranteed to sum
        to ``<= cap_usd``.
        """
        windows = self.windows(start_expiry, end_expiry, window_days=window_days)
        excluded = {str(e) for e in exclude_expiries}
        windows = [w for w in windows if w.expiry not in excluded]
        windows.sort(key=lambda w: w.expiry, reverse=(order == "newest"))
        clamp_start = self._clamp_start(clamp_to_dataset)

        selected: List[dict[str, Any]] = []
        deferred: List[dict[str, Any]] = []
        skipped: List[dict[str, Any]] = []
        spent = 0.0
        budget_exhausted = False
        for w in windows:
            cost, eff_start = self.contract_cost(w, clamp_start=clamp_start)
            row = {"expiry": w.expiry, "raw_symbol": w.raw_symbol, "cost_usd": cost}
            if eff_start is None:
                skipped.append({**row, "skipped": "before_dataset_start", "cost_usd": 0.0})
                continue
            row["start"] = eff_start.isoformat()
            row["end"] = w.end.isoformat()
            if budget_exhausted or spent + cost > cap_usd:
                budget_exhausted = True
                deferred.append(row)
                continue
            spent += cost
            selected.append(row)
        return {
            "dataset": DATASET,
            "schema": SCHEMA,
            "order": order,
            "cap_usd": cap_usd,
            "rate_usd_per_gb": self.unit_price_usd_per_gb(),
            "excluded_count": len(excluded),
            "skipped_before_dataset_start": len(skipped),
            "planned_cost_usd": spent,
            "deferred_cost_usd": sum(r["cost_usd"] for r in deferred),
            "full_cost_usd": spent + sum(r["cost_usd"] for r in deferred),
            "selected": selected,
            "deferred": deferred,
        }

    def _get_range(self, expiry: str, *, window_days: int) -> Any:
        w_start, w_end = contract_window(expiry, window_days=window_days)
        return self._client.timeseries.get_range(
            dataset=DATASET,
            symbols=[cl_raw_symbol(expiry)],
            schema=SCHEMA,
            stype_in=STYPE_IN,
            start=w_start.isoformat(),
            end=w_end.isoformat(),
        )

    def fetch_bars(self, expiry: str, *, window_days: int = 45) -> List[Bar]:
        """Download one contract's windowed 1m bars (BILLABLE)."""
        store = self._get_range(expiry, window_days=window_days)
        return _df_to_bars(store.to_df(price_type="float"))

    def backfill_contract(
        self,
        lake: "MinuteLake",
        expiry: str,
        *,
        window_days: int = 45,
        unit_price_usd_per_gb: float = 280.0,
    ) -> dict[str, Any]:
        """Fetch one contract (BILLABLE) and upsert into the lake under ``expiry``.

        Returns ``{rows, nbytes, actual_cost_usd}`` where ``actual_cost_usd`` is
        derived from the bytes actually received -- a cross-check against the
        pre-flight ``get_cost`` estimate used to enforce the spend cap.
        """
        store = self._get_range(expiry, window_days=window_days)
        nbytes = int(getattr(store, "nbytes", 0) or 0)
        bars = _df_to_bars(store.to_df(price_type="float"))
        rows = lake.upsert_bars("CL", bars, expiry=expiry)
        return {
            "rows": rows,
            "nbytes": nbytes,
            "actual_cost_usd": nbytes / 1e9 * unit_price_usd_per_gb,
        }


def populated_expiries(lake: "MinuteLake", *, symbol: str = "CL", min_rows: int = 1) -> set[str]:
    """Return the set of ``YYYYMM`` expiries already populated in the lake.

    Used by ``--skip-existing`` so a resumed backfill never re-pays for data
    already stored (streaming requests bill on every call regardless of the
    lake's primary-key dedup).
    """
    conn = lake.connect()
    rows = conn.execute(
        f"""
        SELECT expiry
        FROM {lake.bars_table}
        WHERE symbol = ? AND expiry != 'continuous'
        GROUP BY expiry
        HAVING COUNT(*) >= ?
        """,
        [symbol, int(min_rows)],
    ).fetchall()
    return {str(r[0]) for r in rows}


def _parse_utc(value: str) -> datetime:
    import pandas as pd

    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.to_pydatetime()
