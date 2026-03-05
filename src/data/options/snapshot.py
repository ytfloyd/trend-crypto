"""Snapshot implied vol surfaces from Interactive Brokers.

Connects to TWS/Gateway, fetches live option quotes with Greeks
for a given underlying, and stores structured snapshots in DuckDB.
Snapshots can be loaded into VolSurface objects for research.

Usage:
    collector = IBVolSurfaceCollector(db_path="data/market.duckdb")
    collector.connect()
    surface = collector.snapshot("CL", exchange="NYMEX", sec_type="FUT",
                                 last_trade_date="20260601")
    collector.disconnect()

The snapshot flow:
    1. Qualify the underlying contract → get live price
    2. Load stored option chain (or fetch fresh)
    3. For each expiry, request market data for all strikes
    4. Compute mid IV, forward price, moneyness
    5. Store in vol_surface_snaps table
    6. Return a VolSurface object for immediate use
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Optional

import duckdb
import numpy as np
import polars as pl

from common.logging import get_logger
from volatility.surface import VolSlice, VolSurface
from .schema import OptionsSchema

logger = get_logger("ib_vol_surface")

IB_CLIENT_ID_SNAPSHOT = 11
TICK_TIMEOUT_SECS = 10


class IBVolSurfaceCollector:
    """Snapshot and store implied vol surfaces from IB."""

    def __init__(
        self,
        db_path: str,
        host: str = "127.0.0.1",
        port: int = 7497,
        client_id: int = IB_CLIENT_ID_SNAPSHOT,
        rate_limit_secs: float = 0.1,
    ) -> None:
        self.db_path = db_path
        self.host = host
        self.port = port
        self.client_id = client_id
        self.rate_limit_secs = rate_limit_secs
        self._ib: Any = None
        self._conn = duckdb.connect(db_path)
        OptionsSchema(self._conn).ensure_tables()

    def connect(self, timeout: float = 30.0) -> None:
        from ib_insync import IB
        self._ib = IB()
        self._ib.connect(self.host, self.port, clientId=self.client_id, timeout=timeout)
        logger.info("Connected to IB at %s:%d", self.host, self.port)

    def disconnect(self) -> None:
        if self._ib is not None:
            self._ib.disconnect()

    def snapshot(
        self,
        symbol: str,
        exchange: str = "SMART",
        sec_type: str = "STK",
        currency: str = "USD",
        last_trade_date: str = "",
        max_expiries: int = 8,
        strike_range_pct: float = 0.30,
        min_tte_days: int = 3,
        max_tte_days: int = 180,
    ) -> VolSurface:
        """Take a full vol surface snapshot for one underlying.

        Parameters
        ----------
        symbol : underlying symbol.
        exchange, sec_type, currency, last_trade_date : IB contract params.
        max_expiries : limit number of expiries to snapshot (nearest N).
        strike_range_pct : filter strikes to within ±X% of spot.
        min_tte_days, max_tte_days : expiry window.

        Returns
        -------
        VolSurface populated with VolSlices.
        """
        if self._ib is None:
            raise RuntimeError("Not connected to IB")

        underlying_price = self._get_underlying_price(
            symbol, exchange, sec_type, currency, last_trade_date,
        )
        if underlying_price is None or underlying_price <= 0:
            raise ValueError(f"Could not get price for {symbol}")

        chain_df = self._load_or_fetch_chain(
            symbol, exchange, sec_type, currency, last_trade_date,
        )
        if chain_df.is_empty():
            raise ValueError(f"No option chain for {symbol}")

        now = datetime.now(timezone.utc)
        snap_ts = now

        # Filter by TTE
        chain_df = chain_df.with_columns(
            ((pl.col("expiry").cast(pl.Datetime("us")) - pl.lit(now).cast(pl.Datetime("us")))
             .dt.total_seconds() / 86400.0).alias("tte_days")
        )
        chain_df = chain_df.filter(
            (pl.col("tte_days") >= min_tte_days)
            & (pl.col("tte_days") <= max_tte_days)
        )

        # Filter by strike range
        lo = underlying_price * (1.0 - strike_range_pct)
        hi = underlying_price * (1.0 + strike_range_pct)
        chain_df = chain_df.filter(
            (pl.col("strike") >= lo) & (pl.col("strike") <= hi)
        )

        # Limit expiries
        expiries = sorted(chain_df["expiry"].unique().to_list())[:max_expiries]

        surface = VolSurface(underlying=symbol, snapshot_ts=snap_ts)
        all_records: list[dict[str, Any]] = []

        for expiry in expiries:
            expiry_df = chain_df.filter(pl.col("expiry") == expiry)
            tte_days = float(expiry_df["tte_days"][0])
            tte_years = tte_days / 365.25

            strikes = sorted(expiry_df.filter(pl.col("right") == "C")["strike"].unique().to_list())
            if not strikes:
                continue

            tick_data = self._fetch_option_ticks(
                symbol, expiry, strikes, exchange, sec_type,
                currency, last_trade_date, underlying_price,
            )

            if not tick_data:
                continue

            valid_strikes = []
            valid_ivs = []
            bid_ivs = []
            ask_ivs = []

            for td in tick_data:
                if td["mid_iv"] is not None and td["mid_iv"] > 0.001:
                    valid_strikes.append(td["strike"])
                    valid_ivs.append(td["mid_iv"])
                    bid_ivs.append(td.get("bid_iv") or td["mid_iv"])
                    ask_ivs.append(td.get("ask_iv") or td["mid_iv"])

                    all_records.append({
                        "snap_ts": snap_ts,
                        "underlying": symbol,
                        "expiry": expiry,
                        "tte_years": tte_years,
                        "strike": td["strike"],
                        "moneyness": np.log(td["strike"] / underlying_price),
                        "right": td["right"],
                        "underlying_price": underlying_price,
                        "forward": underlying_price,
                        "mid_price": td.get("mid_price"),
                        "bid_iv": td.get("bid_iv"),
                        "ask_iv": td.get("ask_iv"),
                        "mid_iv": td["mid_iv"],
                        "delta": td.get("delta"),
                        "gamma": td.get("gamma"),
                        "vega": td.get("vega"),
                        "open_interest": td.get("open_interest"),
                        "volume": td.get("volume"),
                    })

            if len(valid_strikes) >= 3:
                vol_slice = VolSlice(
                    expiry=expiry if isinstance(expiry, datetime) else datetime.combine(
                        expiry, datetime.min.time(), tzinfo=timezone.utc,
                    ),
                    strikes=np.array(valid_strikes),
                    ivs=np.array(valid_ivs),
                    bid_ivs=np.array(bid_ivs),
                    ask_ivs=np.array(ask_ivs),
                    forward=underlying_price,
                    underlying_price=underlying_price,
                    snapshot_ts=snap_ts,
                )
                surface.add_slice(vol_slice)
                logger.info(
                    "%s %s: %d strikes, ATM IV=%.1f%%",
                    symbol, expiry, len(valid_strikes), vol_slice.atm_iv() * 100,
                )

        if all_records:
            self._store_snapshot(all_records)
            logger.info(
                "%s: stored %d ticks across %d expiries",
                symbol, len(all_records), len(surface.slices),
            )

        return surface

    def _get_underlying_price(
        self,
        symbol: str,
        exchange: str,
        sec_type: str,
        currency: str,
        last_trade_date: str,
    ) -> Optional[float]:
        """Get current price for the underlying."""
        from ib_insync import Stock, Future, Index

        if sec_type == "STK":
            contract = Stock(symbol, exchange, currency)
        elif sec_type == "FUT":
            contract = Future(
                symbol, lastTradeDateOrContractMonth=last_trade_date,
                exchange=exchange, currency=currency,
            )
        elif sec_type == "IND":
            contract = Index(symbol, exchange, currency)
        else:
            return None

        self._ib.qualifyContracts(contract)
        self._ib.reqMktData(contract, "", False, False)
        self._ib.sleep(2)

        ticker = self._ib.ticker(contract)
        if ticker is None:
            return None

        price = ticker.marketPrice()
        self._ib.cancelMktData(contract)

        if price is None or np.isnan(price):
            price = ticker.close
        return float(price) if price and not np.isnan(price) else None

    def _load_or_fetch_chain(
        self,
        symbol: str,
        exchange: str,
        sec_type: str,
        currency: str,
        last_trade_date: str,
    ) -> pl.DataFrame:
        """Load stored chain, or fetch fresh if empty."""
        result = self._conn.execute(
            "SELECT * FROM option_chains WHERE underlying = ? ORDER BY expiry, strike",
            [symbol],
        ).pl()

        if result.is_empty():
            from .chains import IBOptionChainFetcher
            fetcher = IBOptionChainFetcher.__new__(IBOptionChainFetcher)
            fetcher._ib = self._ib
            fetcher._conn = self._conn
            fetcher.db_path = self.db_path
            fetcher.fetch_chain(symbol, exchange, sec_type, currency, last_trade_date)
            result = self._conn.execute(
                "SELECT * FROM option_chains WHERE underlying = ? ORDER BY expiry, strike",
                [symbol],
            ).pl()

        return result

    def _fetch_option_ticks(
        self,
        symbol: str,
        expiry: Any,
        strikes: list[float],
        exchange: str,
        sec_type: str,
        currency: str,
        last_trade_date: str,
        underlying_price: float,
    ) -> list[dict[str, Any]]:
        """Fetch live option quotes for a set of strikes at one expiry."""
        from ib_insync import Option, FuturesOption

        expiry_str = expiry.strftime("%Y%m%d") if hasattr(expiry, "strftime") else str(expiry).replace("-", "")
        results: list[dict[str, Any]] = []

        for strike in strikes:
            for right in ("C", "P"):
                if sec_type == "FUT":
                    contract = FuturesOption(
                        symbol, expiry_str, strike, right,
                        exchange=exchange, currency=currency,
                    )
                else:
                    contract = Option(
                        symbol, expiry_str, strike, right,
                        exchange=exchange, currency=currency,
                    )

                try:
                    self._ib.qualifyContracts(contract)
                except Exception:
                    continue

                self._ib.reqMktData(contract, "", False, False)
                time.sleep(self.rate_limit_secs)

        # Let data arrive
        self._ib.sleep(TICK_TIMEOUT_SECS)

        for strike in strikes:
            for right in ("C", "P"):
                if sec_type == "FUT":
                    contract = FuturesOption(
                        symbol, expiry_str, strike, right,
                        exchange=exchange, currency=currency,
                    )
                else:
                    contract = Option(
                        symbol, expiry_str, strike, right,
                        exchange=exchange, currency=currency,
                    )

                ticker = self._ib.ticker(contract)
                if ticker is None:
                    continue

                bid = ticker.bid if ticker.bid and not np.isnan(ticker.bid) else None
                ask = ticker.ask if ticker.ask and not np.isnan(ticker.ask) else None
                last = ticker.last if ticker.last and not np.isnan(ticker.last) else None

                mid_price = None
                if bid is not None and ask is not None and bid > 0 and ask > 0:
                    mid_price = (bid + ask) / 2.0

                # Use IB's model Greeks if available
                model = ticker.modelGreeks
                ib_iv = model.impliedVol if model and model.impliedVol else None
                ib_delta = model.delta if model and model.delta else None
                ib_gamma = model.gamma if model and model.gamma else None
                ib_vega = model.vega if model and model.vega else None

                # Fall back to our own IV if IB doesn't provide one
                mid_iv = ib_iv

                bid_iv = None
                ask_iv = None
                if ib_iv and mid_price and bid and ask and bid > 0 and ask > 0:
                    spread_ratio = (ask - bid) / mid_price if mid_price > 0 else 0
                    bid_iv = ib_iv * (1 - spread_ratio * 0.5)
                    ask_iv = ib_iv * (1 + spread_ratio * 0.5)

                results.append({
                    "strike": strike,
                    "right": right,
                    "bid": bid,
                    "ask": ask,
                    "last": last,
                    "mid_price": mid_price,
                    "mid_iv": mid_iv,
                    "bid_iv": bid_iv,
                    "ask_iv": ask_iv,
                    "delta": ib_delta,
                    "gamma": ib_gamma,
                    "vega": ib_vega,
                    "volume": int(ticker.volume) if ticker.volume else None,
                    "open_interest": None,
                })

                self._ib.cancelMktData(contract)

        return results

    def _store_snapshot(self, records: list[dict[str, Any]]) -> None:
        """Write snapshot records to vol_surface_snaps."""
        df = pl.DataFrame(records)
        # Delete existing rows for this snap
        snap_ts = records[0]["snap_ts"]
        underlying = records[0]["underlying"]
        self._conn.execute(
            "DELETE FROM vol_surface_snaps WHERE underlying = ? AND snap_ts = ?",
            [underlying, snap_ts],
        )
        self._conn.register("_tmp_snap", df)
        self._conn.execute("""
            INSERT INTO vol_surface_snaps
            SELECT snap_ts, underlying, expiry, tte_years, strike, moneyness,
                   right, underlying_price, forward, mid_price,
                   bid_iv, ask_iv, mid_iv, delta, gamma, vega,
                   open_interest, volume
            FROM _tmp_snap;
        """)
        self._conn.unregister("_tmp_snap")

    def load_surface(
        self,
        underlying: str,
        snap_ts: Optional[datetime] = None,
    ) -> VolSurface:
        """Reconstruct a VolSurface from stored snapshot data.

        If snap_ts is None, loads the most recent snapshot.
        """
        if snap_ts is None:
            row = self._conn.execute(
                "SELECT MAX(snap_ts) FROM vol_surface_snaps WHERE underlying = ?",
                [underlying],
            ).fetchone()
            if not row or row[0] is None:
                raise ValueError(f"No snapshots for {underlying}")
            snap_ts = row[0]

        df = self._conn.execute("""
            SELECT * FROM vol_surface_snaps
            WHERE underlying = ? AND snap_ts = ?
            ORDER BY expiry, strike
        """, [underlying, snap_ts]).pl()

        if df.is_empty():
            raise ValueError(f"No data for {underlying} at {snap_ts}")

        surface = VolSurface(underlying=underlying, snapshot_ts=snap_ts)
        expiries = df["expiry"].unique().sort().to_list()

        for expiry in expiries:
            exp_df = df.filter(
                (pl.col("expiry") == expiry) & (pl.col("right") == "C")
            ).sort("strike")

            if exp_df.is_empty():
                continue

            valid = exp_df.filter(pl.col("mid_iv").is_not_null() & (pl.col("mid_iv") > 0))
            if valid.height < 3:
                continue

            expiry_dt = datetime.combine(
                expiry, datetime.min.time(), tzinfo=timezone.utc,
            ) if not isinstance(expiry, datetime) else expiry

            vol_slice = VolSlice(
                expiry=expiry_dt,
                strikes=valid["strike"].to_numpy(),
                ivs=valid["mid_iv"].to_numpy(),
                bid_ivs=valid["bid_iv"].to_numpy() if "bid_iv" in valid.columns else None,
                ask_ivs=valid["ask_iv"].to_numpy() if "ask_iv" in valid.columns else None,
                forward=float(valid["forward"][0]),
                underlying_price=float(valid["underlying_price"][0]),
                snapshot_ts=snap_ts,
            )
            surface.add_slice(vol_slice)

        return surface

    def list_snapshots(self, underlying: str) -> pl.DataFrame:
        """List all available snapshots for an underlying."""
        return self._conn.execute("""
            SELECT
                snap_ts,
                COUNT(*) AS n_ticks,
                COUNT(DISTINCT expiry) AS n_expiries,
                MIN(strike) AS min_strike,
                MAX(strike) AS max_strike,
                AVG(mid_iv) AS avg_mid_iv
            FROM vol_surface_snaps
            WHERE underlying = ?
            GROUP BY snap_ts
            ORDER BY snap_ts DESC
        """, [underlying]).pl()

    def close(self) -> None:
        self.disconnect()
        self._conn.close()
