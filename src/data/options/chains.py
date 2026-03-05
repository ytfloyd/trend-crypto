"""Fetch option chain definitions from Interactive Brokers.

Connects to TWS/Gateway via ib_insync, discovers available
option contracts for a given underlying, and stores the chain
metadata in DuckDB.

Usage:
    fetcher = IBOptionChainFetcher(db_path="data/market.duckdb")
    fetcher.connect()
    fetcher.fetch_chain("CL", exchange="NYMEX", sec_type="FUT")
    fetcher.fetch_chain("SPY", exchange="SMART", sec_type="STK")
    fetcher.disconnect()

Supports futures options (commodity, rates) and equity/ETF options.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import duckdb
import polars as pl

from common.logging import get_logger
from .schema import OptionsSchema

logger = get_logger("ib_chains")

IB_CLIENT_ID_CHAINS = 10


class IBOptionChainFetcher:
    """Discover and store option chain metadata from IB."""

    def __init__(
        self,
        db_path: str,
        host: str = "127.0.0.1",
        port: int = 7497,
        client_id: int = IB_CLIENT_ID_CHAINS,
    ) -> None:
        self.db_path = db_path
        self.host = host
        self.port = port
        self.client_id = client_id
        self._ib: Any = None
        self._conn = duckdb.connect(db_path)
        OptionsSchema(self._conn).ensure_tables()

    def connect(self, timeout: float = 30.0) -> None:
        """Connect to IB TWS/Gateway.

        Port conventions:
            7497 — TWS paper trading
            7496 — TWS live trading
            4002 — IB Gateway paper
            4001 — IB Gateway live
        """
        from ib_insync import IB
        self._ib = IB()
        self._ib.connect(self.host, self.port, clientId=self.client_id, timeout=timeout)
        logger.info("Connected to IB at %s:%d (client_id=%d)", self.host, self.port, self.client_id)

    def disconnect(self) -> None:
        if self._ib is not None:
            self._ib.disconnect()
            logger.info("Disconnected from IB")

    def fetch_chain(
        self,
        symbol: str,
        exchange: str = "SMART",
        sec_type: str = "STK",
        currency: str = "USD",
        last_trade_date: str = "",
    ) -> int:
        """Fetch the full option chain for an underlying and store in DuckDB.

        Parameters
        ----------
        symbol : underlying symbol (e.g. "CL", "SPY", "ES", "GC").
        exchange : exchange (e.g. "SMART", "NYMEX", "GLOBEX", "COMEX").
        sec_type : "STK" for equity/ETF, "FUT" for futures, "IND" for index.
        currency : typically "USD".
        last_trade_date : for futures, the contract month (e.g. "20260601").

        Returns
        -------
        Number of option contracts found and stored.
        """
        if self._ib is None:
            raise RuntimeError("Not connected to IB. Call connect() first.")

        from ib_insync import Stock, Future, Index

        if sec_type == "STK":
            underlying_contract = Stock(symbol, exchange, currency)
        elif sec_type == "FUT":
            underlying_contract = Future(
                symbol, lastTradeDateOrContractMonth=last_trade_date,
                exchange=exchange, currency=currency,
            )
        elif sec_type == "IND":
            underlying_contract = Index(symbol, exchange, currency)
        else:
            raise ValueError(f"Unsupported sec_type: {sec_type}")

        self._ib.qualifyContracts(underlying_contract)
        chains = self._ib.reqSecDefOptParams(
            underlying_contract.symbol,
            "",
            underlying_contract.secType,
            underlying_contract.conId,
        )

        if not chains:
            logger.warning("No option chains found for %s", symbol)
            return 0

        records: list[dict[str, Any]] = []
        now = datetime.now(timezone.utc)

        for chain in chains:
            chain_exchange = chain.exchange
            multiplier = float(chain.multiplier) if chain.multiplier else 100.0

            for expiry in chain.expirations:
                for strike in chain.strikes:
                    for right in ("C", "P"):
                        records.append({
                            "underlying": symbol,
                            "exchange": chain_exchange,
                            "currency": currency,
                            "expiry": expiry,
                            "strike": float(strike),
                            "right": right,
                            "multiplier": multiplier,
                            "con_id": None,
                            "last_updated": now,
                        })

        if not records:
            logger.warning("Chain for %s has no strikes/expiries", symbol)
            return 0

        df = pl.DataFrame(records)
        logger.info(
            "%s: found %d option contracts across %d expiries",
            symbol, len(records), len(set(r["expiry"] for r in records)),
        )

        self._upsert_chain(df, symbol)
        return len(records)

    def _upsert_chain(self, df: pl.DataFrame, symbol: str) -> None:
        """Replace chain for this underlying and insert fresh data."""
        self._conn.execute(
            "DELETE FROM option_chains WHERE underlying = ?",
            [symbol],
        )
        self._conn.register("_tmp_chains", df)
        self._conn.execute("""
            INSERT INTO option_chains
                (underlying, exchange, currency, expiry, strike, right,
                 multiplier, con_id, last_updated)
            SELECT underlying, exchange, currency, expiry, strike, right,
                   multiplier, con_id, last_updated
            FROM _tmp_chains;
        """)
        self._conn.unregister("_tmp_chains")

    def get_chain(
        self,
        symbol: str,
        min_tte_days: int = 1,
        max_tte_days: int = 365,
    ) -> pl.DataFrame:
        """Load stored chain from DuckDB, filtered by time-to-expiry."""
        return self._conn.execute("""
            SELECT *
            FROM option_chains
            WHERE underlying = ?
              AND expiry >= CURRENT_DATE + INTERVAL ? DAY
              AND expiry <= CURRENT_DATE + INTERVAL ? DAY
            ORDER BY expiry, strike, right
        """, [symbol, min_tte_days, max_tte_days]).pl()

    def list_underlyings(self) -> list[str]:
        """List all underlyings with stored chains."""
        rows = self._conn.execute(
            "SELECT DISTINCT underlying FROM option_chains ORDER BY underlying"
        ).fetchall()
        return [r[0] for r in rows]

    def chain_summary(self, symbol: str) -> dict[str, Any]:
        """Summary stats for a stored chain."""
        row = self._conn.execute("""
            SELECT
                COUNT(*) AS n_contracts,
                COUNT(DISTINCT expiry) AS n_expiries,
                MIN(expiry) AS first_expiry,
                MAX(expiry) AS last_expiry,
                MIN(strike) AS min_strike,
                MAX(strike) AS max_strike,
                MAX(last_updated) AS last_updated
            FROM option_chains
            WHERE underlying = ?
        """, [symbol]).fetchone()
        if not row:
            return {}
        return {
            "underlying": symbol,
            "n_contracts": row[0],
            "n_expiries": row[1],
            "first_expiry": row[2],
            "last_expiry": row[3],
            "min_strike": row[4],
            "max_strike": row[5],
            "last_updated": row[6],
        }

    def close(self) -> None:
        self.disconnect()
        self._conn.close()
