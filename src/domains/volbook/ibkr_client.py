"""Interactive Brokers historical-data client for the volatility book.

Thin wrapper around ``ib_insync`` that qualifies a futures contract and
requests OHLCV bars via ``reqHistoricalData``. Intentionally small: the
CLI drives connect → fetch → disconnect per run so we don't have to
manage a long-lived IB session.

The noise-filter pattern mirrors ``src/data/options/snapshot.py`` — IB
emits chatty warnings that obscure real errors when requesting futures
far from front-month.
"""
from __future__ import annotations

import logging
import re
import time
from datetime import datetime, timezone
from typing import Any

from common.logging import get_logger

from .bundle import Bar, OhlcvSeries
from .contracts import FuturesSpec

logger = get_logger("volbook.ibkr")

IB_CLIENT_ID_DEFAULT = 17
REQ_HIST_TIMEOUT_SECS = 300.0
# An empty response from reqHistoricalData is ambiguous: either IB really has
# no data for that window (= reached data head) or the request timed out and
# ib_insync silently returned ``[]``. We treat any empty response that took at
# least this fraction of the configured timeout as a timeout — real "no data"
# answers come back via Error 162 in well under a second.
_TIMEOUT_DETECTION_RATIO = 0.9


class HistoricalDataTimeout(RuntimeError):
    """``reqHistoricalData`` returned no bars and ran for the full timeout."""
_MONTH_CODE_TO_NUM = {
    "F": "01",
    "G": "02",
    "H": "03",
    "J": "04",
    "K": "05",
    "M": "06",
    "N": "07",
    "Q": "08",
    "U": "09",
    "V": "10",
    "X": "11",
    "Z": "12",
}


class _IBNoiseFilter(logging.Filter):
    """Drop known-noisy ib_insync warnings that bury real errors."""

    _PATTERNS = (
        re.compile(r"Error 2104"),   # market data farm connection is OK
        re.compile(r"Error 2106"),   # HMDS data farm connection is OK
        re.compile(r"Error 2158"),   # sec-def data farm connection is OK
    )

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
        msg = record.getMessage()
        return not any(p.search(msg) for p in self._PATTERNS)


def _install_ib_noise_filter() -> None:
    for name in ("ib_insync.wrapper", "ib_insync.ib"):
        lg = logging.getLogger(name)
        if not any(isinstance(f, _IBNoiseFilter) for f in lg.filters):
            lg.addFilter(_IBNoiseFilter())


def _bar_to_iso(dt: Any) -> str:
    """Normalize ib_insync's ``bar.date`` (date | datetime | str) to ISO."""
    if isinstance(dt, datetime):
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.isoformat(timespec="seconds")
    if hasattr(dt, "isoformat"):
        return dt.isoformat()
    return str(dt)


def _add_months(year_month: str, months: int) -> str:
    """Return ``YYYYMM`` shifted by ``months`` (positive or negative)."""
    if len(year_month) < 6:
        raise ValueError(f"expected YYYYMM, got {year_month!r}")
    year = int(year_month[:4])
    month = int(year_month[4:6])
    total = year * 12 + (month - 1) + months
    new_year, new_month0 = divmod(total, 12)
    return f"{new_year:04d}{new_month0 + 1:02d}"


def _iter_year_months(start: str, end: str) -> list[str]:
    """List every ``YYYYMM`` from ``start`` to ``end`` inclusive."""
    if start > end:
        return []
    months: list[str] = []
    cursor = start
    while cursor <= end:
        months.append(cursor)
        cursor = _add_months(cursor, 1)
    return months


def _contract_month_from_local_symbol(local_symbol: str, *, today: datetime) -> str | None:
    """Infer ``YYYYMM`` from IB local symbols like ``CLM6`` or ``6EZ6``.

    ``reqContractDetails`` often exposes the *last trade date* in
    ``lastTradeDateOrContractMonth`` for commodity futures. For curve
    points we want the contract month instead, which is encoded in the
    local symbol's month code.
    """
    match = re.search(r"([FGHJKMNQUVXZ])(\d)$", local_symbol or "")
    if not match:
        return None
    month_code, year_digit = match.groups()
    month = _MONTH_CODE_TO_NUM[month_code]
    decade = (today.year // 10) * 10
    year = decade + int(year_digit)
    if year < today.year - 1:
        year += 10
    return f"{year}{month}"


class IBHistoricalClient:
    """Connect to TWS/Gateway and pull historical OHLCV for futures."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 7497,
        client_id: int = IB_CLIENT_ID_DEFAULT,
        market_data_type: int = 2,
    ) -> None:
        self.host = host
        self.port = port
        self.client_id = client_id
        self.market_data_type = market_data_type
        self._ib: Any = None

    def connect(self, timeout: float = 30.0) -> None:
        try:
            from ib_insync import IB
        except ImportError as exc:  # pragma: no cover - exercised only without extra
            raise ImportError(
                "ib_insync is required; install with `pip install -e .[options]`"
            ) from exc

        _install_ib_noise_filter()
        self._ib = IB()
        self._ib.connect(self.host, self.port, clientId=self.client_id, timeout=timeout)
        logger.info("Connected to IB at %s:%d (clientId=%d)", self.host, self.port, self.client_id)
        try:
            self._ib.reqMarketDataType(self.market_data_type)
        except Exception as exc:  # noqa: BLE001 - best-effort, IB can be flaky
            logger.warning("reqMarketDataType failed: %s", exc)

    def disconnect(self) -> None:
        if self._ib is not None:
            try:
                self._ib.disconnect()
            finally:
                self._ib = None

    def is_connected(self) -> bool:
        if self._ib is None:
            return False
        try:
            return bool(self._ib.isConnected())
        except Exception:  # noqa: BLE001
            return False

    def reconnect_if_needed(self, *, timeout: float = 30.0) -> bool:
        """Reconnect to IB if the underlying socket has dropped. Returns True if reconnected."""
        if self.is_connected():
            return False
        logger.warning("IB connection dropped; reconnecting %s:%d", self.host, self.port)
        try:
            self.disconnect()
        except Exception:  # noqa: BLE001
            pass
        self.connect(timeout=timeout)
        return True

    def __enter__(self) -> "IBHistoricalClient":
        self.connect()
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.disconnect()

    def _req_minute_bars(
        self,
        contract: Any,
        *,
        end_str: str,
        duration: str,
        what_to_show: str,
        use_rth: bool,
        timeout: float = REQ_HIST_TIMEOUT_SECS,
        label: str | None = None,
    ) -> list[Any]:
        """Issue ``reqHistoricalData`` for 1-min bars and detect timeouts.

        Wraps the call with a wall-clock timer so callers can distinguish
        a genuinely empty response (Error 162: HMDS query returned no
        data, returned in well under a second) from a timeout (where
        ib_insync silently returns ``[]`` after ``timeout`` seconds).

        Raises :class:`HistoricalDataTimeout` on the latter so the caller
        can retry rather than mistake a timeout for the head of data.
        """
        started = time.monotonic()
        raw_bars = self._ib.reqHistoricalData(
            contract,
            endDateTime=end_str,
            durationStr=duration,
            barSizeSetting="1 min",
            whatToShow=what_to_show,
            useRTH=use_rth,
            formatDate=2,
            timeout=timeout,
        )
        elapsed = time.monotonic() - started
        if not raw_bars and elapsed >= timeout * _TIMEOUT_DETECTION_RATIO:
            tag = label or getattr(contract, "localSymbol", None) or "contract"
            raise HistoricalDataTimeout(
                f"reqHistoricalData timed out after {elapsed:.1f}s for {tag} "
                f"(end={end_str or 'now'}, duration={duration})"
            )
        return list(raw_bars or [])

    @staticmethod
    def _future_kwargs(spec: FuturesSpec) -> dict[str, str]:
        contract_kwargs = {}
        if spec.multiplier:
            contract_kwargs["multiplier"] = spec.multiplier
        if spec.trading_class:
            contract_kwargs["tradingClass"] = spec.trading_class
        return contract_kwargs

    def discover_futures_curve(
        self,
        spec: FuturesSpec,
        *,
        limit: int | None = 5,
        min_expiry: str | None = None,
    ) -> list[FuturesSpec]:
        """Discover the first ``limit`` active futures expiries for a root.

        IB returns contract months as either ``YYYYMM`` or full last-trade
        dates (``YYYYMMDD``). We filter out already-expired contracts,
        dedupe by contract month, then return specs sorted by expiry.
        """
        if self._ib is None:
            raise RuntimeError("not connected; call connect() first")

        from ib_insync import Future

        today = datetime.now(timezone.utc)
        cutoff = min_expiry or today.strftime("%Y%m%d")
        current_month = today.strftime("%Y%m")
        root = Future(
            symbol=spec.symbol,
            exchange=spec.exchange,
            currency=spec.currency,
            **self._future_kwargs(spec),
        )
        details = self._ib.reqContractDetails(root)
        if not details:
            raise RuntimeError(
                f"IB returned no contract details for {spec.label_symbol} on {spec.exchange}"
            )

        months: set[str] = set()
        for detail in details:
            contract = detail.contract
            expiry = str(getattr(contract, "lastTradeDateOrContractMonth", "") or "")
            if len(expiry) < 6:
                continue
            # Keep active maturities. If IB gives only YYYYMM, compare to the
            # current month; if it gives YYYYMMDD, compare to today.
            comparable = expiry if len(expiry) >= 8 else f"{expiry}99"
            if comparable < cutoff:
                continue
            local_symbol = str(getattr(contract, "localSymbol", "") or "")
            contract_month = _contract_month_from_local_symbol(local_symbol, today=today)
            month = contract_month or expiry[:6]
            # Prefer forward curve points: current-month contracts are often
            # stale/near expiry and IB may list them in contract details while
            # refusing historical qualification by YYYYMM.
            if month <= current_month:
                continue
            months.add(month)

        expiries = sorted(months)
        if limit and limit > 0:
            expiries = expiries[:limit]
        if not expiries:
            raise RuntimeError(f"no active futures expiries found for {spec.label_symbol}")

        curve = [spec.with_expiry(expiry) for expiry in expiries]
        logger.info(
            "Discovered %s curve: %s",
            spec.label_symbol,
            ", ".join(s.label for s in curve),
        )
        return curve

    def fetch_futures_ohlcv(
        self,
        spec: FuturesSpec,
        *,
        bar_size: str = "1 day",
        duration: str = "1 Y",
        what_to_show: str = "TRADES",
        use_rth: bool = False,
        end_datetime: str = "",
    ) -> OhlcvSeries:
        """Pull historical OHLCV for ``spec`` and return a populated series.

        Parameters follow IB ``reqHistoricalData`` conventions: ``bar_size``
        e.g. ``"1 day"``, ``"1 hour"``, ``"5 mins"``; ``duration`` e.g.
        ``"1 Y"``, ``"6 M"``, ``"30 D"``; ``what_to_show`` typically
        ``TRADES``, ``MIDPOINT``, or ``BID_ASK``.
        """
        if self._ib is None:
            raise RuntimeError("not connected; call connect() first")

        from ib_insync import Future

        contract = Future(
            symbol=spec.symbol,
            lastTradeDateOrContractMonth=spec.expiry,
            exchange=spec.exchange,
            currency=spec.currency,
            **self._future_kwargs(spec),
        )
        qualified = self._ib.qualifyContracts(contract)
        if not qualified or qualified[0].conId == 0:
            raise RuntimeError(
                f"could not qualify {spec.symbol} {spec.expiry} on {spec.exchange}"
            )
        contract = qualified[0]
        logger.info(
            "Qualified %s conId=%d localSymbol=%s",
            spec.label, contract.conId, getattr(contract, "localSymbol", ""),
        )

        raw_bars = self._ib.reqHistoricalData(
            contract,
            endDateTime=end_datetime,
            durationStr=duration,
            barSizeSetting=bar_size,
            whatToShow=what_to_show,
            useRTH=use_rth,
            formatDate=2,  # UTC
            timeout=REQ_HIST_TIMEOUT_SECS,
        )
        if not raw_bars:
            raise RuntimeError(
                f"IB returned no bars for {spec.label} ({bar_size}, {duration})"
            )

        bars = [
            Bar(
                t=_bar_to_iso(b.date),
                o=float(b.open),
                h=float(b.high),
                l=float(b.low),
                c=float(b.close),
                v=float(b.volume),
            )
            for b in raw_bars
        ]
        logger.info("Pulled %d bars for %s %s", len(bars), spec.label, bar_size)

        return OhlcvSeries(
            symbol=spec.label_symbol,
            expiry=spec.expiry,
            exchange=spec.exchange,
            bar_size=bar_size,
            duration=duration,
            what_to_show=what_to_show,
            bars=bars,
            fetched_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            currency=spec.currency,
        )

    def _qualify_continuous(self, spec: FuturesSpec) -> Any:
        from ib_insync import ContFuture

        contract = ContFuture(
            symbol=spec.symbol,
            exchange=spec.exchange,
            currency=spec.currency,
            **self._future_kwargs(spec),
        )
        qualified = self._ib.qualifyContracts(contract)
        if not qualified or qualified[0].conId == 0:
            raise RuntimeError(
                f"could not qualify continuous {spec.label_symbol} on {spec.exchange}"
            )
        contract = qualified[0]
        logger.info(
            "Qualified %s continuous conId=%d localSymbol=%s",
            spec.label_symbol,
            contract.conId,
            getattr(contract, "localSymbol", ""),
        )
        return contract

    def head_timestamp_continuous(
        self,
        spec: FuturesSpec,
        *,
        what_to_show: str = "TRADES",
        use_rth: bool = False,
    ) -> datetime | None:
        """Return IBKR's earliest available timestamp for a continuous spec."""
        if self._ib is None:
            raise RuntimeError("not connected; call connect() first")
        contract = self._qualify_continuous(spec)
        head = self._ib.reqHeadTimeStamp(
            contract,
            whatToShow=what_to_show,
            useRTH=use_rth,
            formatDate=2,
        )
        if head in (None, "", 0):
            return None
        if isinstance(head, datetime):
            return head if head.tzinfo else head.replace(tzinfo=timezone.utc)
        try:
            text = str(head)
            if text.endswith("Z"):
                text = text[:-1] + "+00:00"
            dt = datetime.fromisoformat(text)
        except ValueError:
            logger.warning("Unparseable head timestamp for %s: %r", spec.label_symbol, head)
            return None
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)

    def fetch_continuous_futures_ohlcv(
        self,
        spec: FuturesSpec,
        *,
        bar_size: str = "1 day",
        duration: str = "1 Y",
        what_to_show: str = "TRADES",
        use_rth: bool = False,
        end_datetime: str = "",
    ) -> OhlcvSeries:
        """Pull historical OHLCV for IB's front/continuous futures contract."""
        if self._ib is None:
            raise RuntimeError("not connected; call connect() first")

        contract = self._qualify_continuous(spec)
        raw_bars = self._ib.reqHistoricalData(
            contract,
            endDateTime=end_datetime,
            durationStr=duration,
            barSizeSetting=bar_size,
            whatToShow=what_to_show,
            useRTH=use_rth,
            formatDate=2,  # UTC
            timeout=REQ_HIST_TIMEOUT_SECS,
        )
        if not raw_bars:
            raise RuntimeError(
                f"IB returned no bars for {spec.label_symbol} continuous ({bar_size}, {duration})"
            )

        bars = [
            Bar(
                t=_bar_to_iso(b.date),
                o=float(b.open),
                h=float(b.high),
                l=float(b.low),
                c=float(b.close),
                v=float(b.volume),
            )
            for b in raw_bars
        ]
        logger.info(
            "Pulled %d bars for %s continuous %s",
            len(bars),
            spec.label_symbol,
            bar_size,
        )

        return OhlcvSeries(
            symbol=spec.label_symbol,
            expiry="continuous",
            exchange=spec.exchange,
            bar_size=bar_size,
            duration=duration,
            what_to_show=what_to_show,
            bars=bars,
            fetched_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            currency=spec.currency,
        )

    def fetch_continuous_minute_bars(
        self,
        spec: FuturesSpec,
        *,
        duration: str = "30 D",
        what_to_show: str = "TRADES",
        use_rth: bool = False,
    ) -> list[Bar]:
        """Return raw 1-minute bars for ``spec``'s continuous future.

        IBKR rejects ``endDateTime`` on ``ContFuture`` historical requests
        (error 10339 — and follows up with a socket disconnect), so this
        method always asks for the *trailing* window of size ``duration``.
        IB also paces 1-minute bar requests aggressively; for ContFuture the
        practical max ``duration`` is around ``"30 D"`` per call.
        """
        if self._ib is None:
            raise RuntimeError("not connected; call connect() first")

        contract = self._qualify_continuous(spec)
        raw_bars = self._req_minute_bars(
            contract,
            end_str="",
            duration=duration,
            what_to_show=what_to_show,
            use_rth=use_rth,
            label=f"{spec.label_symbol} continuous",
        )
        if not raw_bars:
            return []
        bars = [
            Bar(
                t=_bar_to_iso(b.date),
                o=float(b.open),
                h=float(b.high),
                l=float(b.low),
                c=float(b.close),
                v=float(b.volume),
            )
            for b in raw_bars
        ]
        logger.info(
            "Pulled %d trailing 1-min bars for %s continuous (duration=%s)",
            len(bars),
            spec.label_symbol,
            duration,
        )
        return bars

    def discover_dated_futures(
        self,
        spec: FuturesSpec,
        *,
        min_expiry: str | None = None,
        max_expiry: str | None = None,
        include_expired: bool = True,
    ) -> list[FuturesSpec]:
        """Return dated ``FuturesSpec``s known to IB for ``spec``'s root.

        For each candidate ``YYYYMM`` between ``min_expiry`` and
        ``max_expiry`` (inclusive), issues a *targeted*
        ``reqContractDetails`` for that specific contract month with
        ``includeExpired``. This is dramatically more reliable than a
        single bare ``includeExpired=True`` call, which IB will accept
        but often never finalize (it tries to return every contract
        that ever existed for the root, including options/spreads).

        Defaults: ``min_expiry`` = current month, ``max_expiry`` = 24
        months out, so the call is bounded even if the caller forgets
        to set a window. Returned specs are sorted ascending by expiry.
        """
        if self._ib is None:
            raise RuntimeError("not connected; call connect() first")

        from ib_insync import Future

        today = datetime.now(timezone.utc)
        floor = min_expiry or f"{today.year:04d}{today.month:02d}"
        ceiling = max_expiry or _add_months(floor, 24)
        if floor > ceiling:
            return []

        candidates = _iter_year_months(floor, ceiling)
        months: set[str] = set()
        for year_month in candidates:
            probe = Future(
                symbol=spec.symbol,
                exchange=spec.exchange,
                currency=spec.currency,
                lastTradeDateOrContractMonth=year_month,
                includeExpired=include_expired,
                **self._future_kwargs(spec),
            )
            try:
                details = self._ib.reqContractDetails(probe)
            except Exception as exc:  # noqa: BLE001
                logger.debug(
                    "reqContractDetails failed for %s %s: %s",
                    spec.label_symbol, year_month, exc,
                )
                continue
            if not details:
                continue
            for detail in details:
                contract = detail.contract
                expiry = str(getattr(contract, "lastTradeDateOrContractMonth", "") or "")
                if len(expiry) < 6:
                    continue
                month = expiry[:6]
                if month < floor or month > ceiling:
                    continue
                months.add(month)

        ordered = sorted(months)
        logger.info(
            "Discovered %d dated %s expiries in [%s, %s]",
            len(ordered),
            spec.label_symbol,
            floor,
            ceiling,
        )
        return [spec.with_expiry(m) for m in ordered]

    def qualify_dated_futures(
        self,
        specs: list[FuturesSpec],
    ) -> list[tuple[FuturesSpec, Any | None]]:
        """Batch-qualify a list of dated futures specs in one IB round trip.

        Returns ``[(spec, contract_or_None), ...]`` in the same order as
        ``specs``. ``contract`` is ``None`` for specs that don't resolve
        to a listed IB contract (e.g. month codes outside the root's
        cycle, or expiries IB no longer has metadata for).

        Uses ``ib_insync.IB.qualifyContracts`` which sends all
        ``reqContractDetails`` requests in parallel and waits for every
        ``contractDetailsEnd`` callback. One TCP round trip for the
        whole batch — far cheaper than N serial qualifies.
        """
        if self._ib is None:
            raise RuntimeError("not connected; call connect() first")
        if not specs:
            return []

        from ib_insync import Future

        contracts: list[Any] = []
        for spec in specs:
            if not spec.expiry:
                raise ValueError("qualify_dated_futures requires spec.expiry on each spec")
            contracts.append(
                Future(
                    symbol=spec.symbol,
                    lastTradeDateOrContractMonth=spec.expiry,
                    exchange=spec.exchange,
                    currency=spec.currency,
                    includeExpired=True,
                    **self._future_kwargs(spec),
                )
            )

        try:
            self._ib.qualifyContracts(*contracts)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Batch qualifyContracts failed for %s: %s; falling back to per-spec",
                specs[0].label_symbol,
                exc,
            )
            return [(spec, None) for spec in specs]

        results: list[tuple[FuturesSpec, Any | None]] = []
        for spec, contract in zip(specs, contracts):
            if getattr(contract, "conId", 0):
                results.append((spec, contract))
            else:
                results.append((spec, None))
        listed = sum(1 for _, c in results if c is not None)
        logger.info(
            "Qualified %s: %d/%d dated candidates exist",
            specs[0].label_symbol,
            listed,
            len(specs),
        )
        return results

    def fetch_dated_minute_bars_by_contract(
        self,
        contract: Any,
        *,
        end_datetime: datetime | None = None,
        duration: str = "30 D",
        what_to_show: str = "TRADES",
        use_rth: bool = False,
    ) -> list[Bar]:
        """Fetch 1-min bars for an *already-qualified* dated contract.

        Skips the qualifier round trip — caller passes the contract
        returned by :meth:`qualify_dated_futures`. Use this in tight
        loops over many chunks for the same contract.
        """
        if self._ib is None:
            raise RuntimeError("not connected; call connect() first")

        if end_datetime is None:
            end_str = ""
        else:
            end = end_datetime.astimezone(timezone.utc)
            end_str = end.strftime("%Y%m%d %H:%M:%S UTC")

        raw_bars = self._req_minute_bars(
            contract,
            end_str=end_str,
            duration=duration,
            what_to_show=what_to_show,
            use_rth=use_rth,
            label=getattr(contract, "localSymbol", None),
        )
        if not raw_bars:
            return []
        bars = [
            Bar(
                t=_bar_to_iso(b.date),
                o=float(b.open),
                h=float(b.high),
                l=float(b.low),
                c=float(b.close),
                v=float(b.volume),
            )
            for b in raw_bars
        ]
        return bars

    def fetch_dated_minute_bars_with_retry(
        self,
        contract: Any,
        *,
        end_datetime: datetime | None = None,
        duration: str = "30 D",
        what_to_show: str = "TRADES",
        use_rth: bool = False,
        max_retries: int = 3,
        backoff_seconds: float = 15.0,
        label: str | None = None,
    ) -> list[Bar]:
        """Fetch a 1-min chunk and retry on :class:`HistoricalDataTimeout`.

        Real "no more data" responses come back as ``[]`` in well under a
        second; the underlying fetcher raises
        :class:`HistoricalDataTimeout` on stuck requests so callers can
        retry rather than mistake a timeout for the head/tail of data.

        On a timeout we sleep ``backoff_seconds`` then attempt to
        reconnect before retrying. After ``max_retries`` attempts the
        timeout is re-raised so the caller can record a failure instead
        of advancing past missing data.
        """
        tag = label or getattr(contract, "localSymbol", None) or "contract"
        last_exc: HistoricalDataTimeout | None = None
        for attempt in range(1, max_retries + 1):
            try:
                return self.fetch_dated_minute_bars_by_contract(
                    contract,
                    end_datetime=end_datetime,
                    duration=duration,
                    what_to_show=what_to_show,
                    use_rth=use_rth,
                )
            except HistoricalDataTimeout as exc:
                last_exc = exc
                if attempt < max_retries:
                    logger.warning(
                        "%s: chunk timed out (attempt %d/%d): %s; "
                        "sleeping %.0fs then reconnecting",
                        tag,
                        attempt,
                        max_retries,
                        exc,
                        backoff_seconds,
                    )
                    time.sleep(backoff_seconds)
                    try:
                        self.reconnect_if_needed()
                    except Exception:  # noqa: BLE001
                        pass
                else:
                    logger.warning(
                        "%s: chunk timed out %d times in a row; bailing",
                        tag,
                        max_retries,
                    )
        assert last_exc is not None  # for type-checkers
        raise last_exc

    def head_timestamp_dated(
        self,
        spec: FuturesSpec,
        *,
        what_to_show: str = "TRADES",
        use_rth: bool = False,
    ) -> datetime | None:
        """Return IBKR's earliest available timestamp for a dated contract."""
        if self._ib is None:
            raise RuntimeError("not connected; call connect() first")
        if not spec.expiry:
            raise ValueError("head_timestamp_dated requires spec.expiry")

        from ib_insync import Future

        contract = Future(
            symbol=spec.symbol,
            lastTradeDateOrContractMonth=spec.expiry,
            exchange=spec.exchange,
            currency=spec.currency,
            includeExpired=True,
            **self._future_kwargs(spec),
        )
        qualified = self._ib.qualifyContracts(contract)
        if not qualified or qualified[0].conId == 0:
            raise RuntimeError(
                f"could not qualify dated {spec.symbol} {spec.expiry} on {spec.exchange}"
            )
        contract = qualified[0]

        head = self._ib.reqHeadTimeStamp(
            contract,
            whatToShow=what_to_show,
            useRTH=use_rth,
            formatDate=2,
        )
        if head in (None, "", 0):
            return None
        if isinstance(head, datetime):
            return head if head.tzinfo else head.replace(tzinfo=timezone.utc)
        try:
            text = str(head)
            if text.endswith("Z"):
                text = text[:-1] + "+00:00"
            dt = datetime.fromisoformat(text)
        except ValueError:
            logger.warning("Unparseable head timestamp for %s: %r", spec.label, head)
            return None
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)

    def fetch_dated_minute_bars(
        self,
        spec: FuturesSpec,
        *,
        end_datetime: datetime | None = None,
        duration: str = "30 D",
        what_to_show: str = "TRADES",
        use_rth: bool = False,
    ) -> list[Bar]:
        """Return raw 1-minute bars for a *dated* futures contract.

        Used by the deep-history dated-walk path. Unlike ``ContFuture``,
        dated ``Future`` contracts accept ``endDateTime``, so the caller
        can chunk backward through history.
        """
        if self._ib is None:
            raise RuntimeError("not connected; call connect() first")
        if not spec.expiry:
            raise ValueError("fetch_dated_minute_bars requires spec.expiry")

        from ib_insync import Future

        contract = Future(
            symbol=spec.symbol,
            lastTradeDateOrContractMonth=spec.expiry,
            exchange=spec.exchange,
            currency=spec.currency,
            includeExpired=True,
            **self._future_kwargs(spec),
        )
        qualified = self._ib.qualifyContracts(contract)
        if not qualified or qualified[0].conId == 0:
            raise RuntimeError(
                f"could not qualify dated {spec.symbol} {spec.expiry} on {spec.exchange}"
            )
        contract = qualified[0]

        if end_datetime is None:
            end_str = ""
        else:
            end = end_datetime.astimezone(timezone.utc)
            end_str = end.strftime("%Y%m%d %H:%M:%S UTC")

        raw_bars = self._req_minute_bars(
            contract,
            end_str=end_str,
            duration=duration,
            what_to_show=what_to_show,
            use_rth=use_rth,
            label=spec.label,
        )
        if not raw_bars:
            return []
        bars = [
            Bar(
                t=_bar_to_iso(b.date),
                o=float(b.open),
                h=float(b.high),
                l=float(b.low),
                c=float(b.close),
                v=float(b.volume),
            )
            for b in raw_bars
        ]
        logger.info(
            "Pulled %d 1-min bars for dated %s ending %s",
            len(bars),
            spec.label,
            end_str or "now",
        )
        return bars
