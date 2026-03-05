"""
On-chain data fetcher for Alpha Lab.

Fetches BTC on-chain metrics from blockchain.com (free, no API key),
caches to parquet, and provides a clean daily DataFrame aligned to
the trading calendar.

Metrics fetched:
  - hash_rate: Network hash rate (TH/s)
  - n_transactions: Confirmed transactions per day
  - tx_volume_usd: Estimated USD transaction value
  - miners_revenue: Total miner revenue (USD)
  - n_unique_addresses: Number of unique addresses used
  - difficulty: Mining difficulty
  - mempool_size: Mempool size (bytes)
  - tx_fees_usd: Total transaction fees (USD)
  - cost_per_tx: Average cost per transaction (USD)
  - n_tx_per_block: Average transactions per block
  - output_volume: Total output volume (BTC)
  - total_btc: Total bitcoins in circulation
  - market_cap: BTC market cap (USD)
  - utxo_count: Number of unspent transaction outputs
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

_CACHE_DIR = Path(__file__).resolve().parent / "_cache"

BLOCKCHAIN_CHARTS = {
    "hash_rate": "hash-rate",
    "n_transactions": "n-transactions",
    "tx_volume_usd": "estimated-transaction-volume-usd",
    "miners_revenue": "miners-revenue",
    "n_unique_addresses": "n-unique-addresses",
    "difficulty": "difficulty",
    "mempool_size": "mempool-size",
    "tx_fees_usd": "transaction-fees-usd",
    "cost_per_tx": "cost-per-transaction",
    "n_tx_per_block": "n-transactions-per-block",
    "output_volume": "output-volume",
    "total_btc": "total-bitcoins",
    "market_cap": "market-cap",
    "utxo_count": "utxo-count",
}

_BASE_URL = "https://api.blockchain.info/charts"


def _fetch_chart_chunked(chart_name: str, start_year: int = 2017) -> pd.Series:
    """Fetch a blockchain.com chart in 2-year chunks to get daily granularity."""
    all_points: dict[str, float] = {}
    current = datetime(start_year, 1, 1)
    end = datetime.now()

    while current < end:
        chunk_end = min(current + timedelta(days=729), end)
        timespan = f"{(chunk_end - current).days}days"
        start_str = current.strftime("%Y-%m-%d")

        url = (
            f"{_BASE_URL}/{chart_name}"
            f"?timespan={timespan}&start={start_str}"
            f"&format=json&rollingAverage=1days"
        )
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            data = r.json()
            for pt in data.get("values", []):
                dt = datetime.utcfromtimestamp(pt["x"]).strftime("%Y-%m-%d")
                all_points[dt] = pt["y"]
        except Exception as e:
            print(f"  [onchain] Warning: {chart_name} chunk {start_str}: {e}")

        current = chunk_end + timedelta(days=1)
        time.sleep(0.3)

    if not all_points:
        return pd.Series(dtype=float)

    s = pd.Series(all_points, dtype=float)
    s.index = pd.to_datetime(s.index)
    s = s.sort_index()
    s = s[~s.index.duplicated(keep="last")]
    return s


def fetch_all_onchain(
    start: str = "2017-01-01",
    end: str = "2025-12-15",
    use_cache: bool = True,
) -> pd.DataFrame:
    """Fetch all on-chain metrics and return as a daily DataFrame.

    Returns DataFrame with DatetimeIndex and one column per metric.
    Missing days are forward-filled (on-chain data can have gaps on weekends).
    """
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = _CACHE_DIR / "onchain_btc_daily.parquet"

    if use_cache and cache_path.exists():
        df = pd.read_parquet(cache_path)
        df = df.loc[start:end]
        if len(df) > 100:
            print(f"[onchain] Loaded cached data: {len(df)} rows, {len(df.columns)} metrics")
            return df

    start_year = int(start[:4])
    print(f"[onchain] Fetching BTC on-chain data from blockchain.com ...")
    frames: dict[str, pd.Series] = {}

    for col_name, chart_name in BLOCKCHAIN_CHARTS.items():
        print(f"  [onchain] Fetching {col_name} ({chart_name}) ...")
        s = _fetch_chart_chunked(chart_name, start_year=start_year)
        if len(s) > 0:
            frames[col_name] = s
            print(f"    -> {len(s)} daily points")
        else:
            print(f"    -> FAILED (no data)")

    if not frames:
        print("[onchain] WARNING: No on-chain data fetched!")
        return pd.DataFrame()

    df = pd.DataFrame(frames)
    df.index.name = "date"

    date_range = pd.date_range(start=start, end=end, freq="D")
    df = df.reindex(date_range)
    df = df.ffill(limit=7)

    df.to_parquet(cache_path)
    print(f"[onchain] Cached to {cache_path}: {len(df)} rows, {len(df.columns)} metrics")
    return df


def compute_derived_onchain(df: pd.DataFrame) -> pd.DataFrame:
    """Compute derived on-chain metrics from raw data.

    These are the actual alpha-relevant features, not the raw metrics.
    """
    out = pd.DataFrame(index=df.index)

    # NVT Ratio: Market Cap / Transaction Volume (Network Value to Transactions)
    # High NVT = overvalued network, low NVT = undervalued
    if "market_cap" in df.columns and "tx_volume_usd" in df.columns:
        out["nvt_ratio"] = df["market_cap"] / df["tx_volume_usd"].replace(0, float("nan"))
        out["nvt_ratio_28d"] = out["nvt_ratio"].rolling(28).median()

    # Hash Rate Momentum: rate of change of hash rate = miner confidence
    if "hash_rate" in df.columns:
        hr = df["hash_rate"]
        out["hash_rate_mom_7d"] = hr / hr.shift(7) - 1.0
        out["hash_rate_mom_30d"] = hr / hr.shift(30) - 1.0
        out["hash_rate_z"] = (hr - hr.rolling(90).mean()) / hr.rolling(90).std()

    # Difficulty Ribbon: ratio of short to long difficulty MA
    # Compressed ribbon = miner capitulation = historically bullish
    if "difficulty" in df.columns:
        diff = df["difficulty"]
        out["diff_ribbon"] = diff.rolling(9).mean() / diff.rolling(200).mean()
        out["diff_ribbon_z"] = (
            (out["diff_ribbon"] - out["diff_ribbon"].rolling(90).mean())
            / out["diff_ribbon"].rolling(90).std()
        )

    # Transaction Count Momentum
    if "n_transactions" in df.columns:
        tx = df["n_transactions"]
        out["tx_count_mom_7d"] = tx / tx.shift(7) - 1.0
        out["tx_count_mom_30d"] = tx / tx.shift(30) - 1.0
        out["tx_count_z"] = (tx - tx.rolling(90).mean()) / tx.rolling(90).std()

    # Active Addresses Momentum
    if "n_unique_addresses" in df.columns:
        addr = df["n_unique_addresses"]
        out["active_addr_mom_7d"] = addr / addr.shift(7) - 1.0
        out["active_addr_mom_30d"] = addr / addr.shift(30) - 1.0
        out["active_addr_z"] = (addr - addr.rolling(90).mean()) / addr.rolling(90).std()

    # Fee Pressure: high fees = network congestion = demand
    if "tx_fees_usd" in df.columns:
        fees = df["tx_fees_usd"]
        out["fee_pressure_z"] = (fees - fees.rolling(30).mean()) / fees.rolling(30).std()
        out["fee_ratio"] = fees / fees.rolling(90).mean()

    # Miner Revenue per Hash (efficiency): declining = miner stress
    if "miners_revenue" in df.columns and "hash_rate" in df.columns:
        out["revenue_per_hash"] = df["miners_revenue"] / df["hash_rate"].replace(0, float("nan"))
        rph = out["revenue_per_hash"]
        out["revenue_per_hash_z"] = (rph - rph.rolling(90).mean()) / rph.rolling(90).std()

    # Mempool Congestion: high mempool = high demand/urgency
    if "mempool_size" in df.columns:
        mem = df["mempool_size"]
        out["mempool_z"] = (mem - mem.rolling(30).mean()) / mem.rolling(30).std()

    # UTXO Count Change: growing UTXO set = more holders (HODLing)
    if "utxo_count" in df.columns:
        utxo = df["utxo_count"]
        out["utxo_growth_7d"] = utxo / utxo.shift(7) - 1.0
        out["utxo_growth_30d"] = utxo / utxo.shift(30) - 1.0

    # Supply Inflation Rate: new BTC minted / total supply
    if "total_btc" in df.columns:
        total = df["total_btc"]
        out["supply_inflation_30d"] = (total - total.shift(30)) / total.shift(30)

    # Transaction Volume / Market Cap Ratio (inverse NVT, velocity)
    if "tx_volume_usd" in df.columns and "market_cap" in df.columns:
        out["velocity"] = df["tx_volume_usd"] / df["market_cap"].replace(0, float("nan"))
        out["velocity_z"] = (
            (out["velocity"] - out["velocity"].rolling(90).mean())
            / out["velocity"].rolling(90).std()
        )

    # Output Volume Momentum (BTC moved on-chain)
    if "output_volume" in df.columns:
        ov = df["output_volume"]
        out["output_vol_mom_7d"] = ov / ov.shift(7) - 1.0
        out["output_vol_z"] = (ov - ov.rolling(30).mean()) / ov.rolling(30).std()

    # Cost per Transaction Regime
    if "cost_per_tx" in df.columns:
        cpt = df["cost_per_tx"]
        out["cost_per_tx_z"] = (cpt - cpt.rolling(90).mean()) / cpt.rolling(90).std()

    return out.ffill(limit=3)
