"""
Medallion Lite — live signal service.

Stateful service that computes target weights each cycle (hourly) by:
  1. Loading recent OHLCV from DuckDB
  2. Computing ensemble regime score
  3. Computing cross-sectional factor model
  4. Maintaining portfolio holdings state (entries, exits, trailing stops)
  5. Publishing target weights to a well-defined contract

Designed to run as a cron job (every hour) or embedded in a scheduler.
State persists between runs via a JSON file.

Architecture:
  [Collector] → [DuckDB] → [SignalService] → [signal_output.json]
                                                      ↓
                                            [Execution Engine]
"""
from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import duckdb
import numpy as np
import pandas as pd

logger = logging.getLogger("medallion_signal")

ANN_FACTOR = 8760.0
DEFAULT_DB = str(
    Path(__file__).resolve().parents[2] / ".." / "data" / "market.duckdb"
)

# ── Signal Output Contract ──────────────────────────────────────────


@dataclass
class TradeAction:
    """Single entry or exit action for the execution engine."""
    symbol: str
    action: str          # entry | exit_regime | exit_stop | exit_factor | exit_maxhold
    target_weight: float
    reason: str = ""
    hours_held: int = 0
    cum_ret: float = 0.0
    score: float = 0.0


@dataclass
class SignalOutput:
    """The contract between signal service and execution engine.

    This is what gets published as JSON and/or written to the
    live_signals DuckDB table. The execution engine reads this
    and calls OMS.rebalance_to_targets(target_weights, ...).
    """
    ts: str                              # ISO 8601 UTC
    cycle_id: str                        # UUID for idempotency
    target_weights: dict[str, float]     # symbol → weight [0, 1]
    regime_score: float                  # [0, 1]
    actions: list[dict[str, Any]]        # list of TradeAction dicts
    diagnostics: dict[str, Any]          # n_holdings, gross_exposure, etc.
    stale: bool = False                  # True if data freshness check failed


# ── Holding State ───────────────────────────────────────────────────


@dataclass
class Holding:
    """State for a single open position."""
    symbol: str
    entry_ts: str
    entry_score: float
    hours_held: int = 0
    cum_ret: float = 0.0
    peak_cum: float = 0.0


@dataclass
class PortfolioState:
    """Persistent state between signal service cycles."""
    holdings: dict[str, dict[str, Any]] = field(default_factory=dict)
    last_cycle_ts: str = ""
    last_rebalance_hour: int = 0
    cycle_count: int = 0

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, default=str))

    @classmethod
    def load(cls, path: Path) -> PortfolioState:
        if path.exists():
            data = json.loads(path.read_text())
            return cls(**data)
        return cls()


# ── Signal Service ──────────────────────────────────────────────────


@dataclass
class SignalConfig:
    """Configuration for the Medallion signal service."""
    db_path: str = DEFAULT_DB
    state_path: str = "medallion_state.json"
    output_path: str = "signal_output.json"
    output_db_table: str = "live_signals"

    # Universe
    min_adv_usd: float = 5_000_000
    max_symbols: int = 50
    min_history_hours: int = 720

    # Regime weights
    w_trend: float = 0.40
    w_breadth: float = 0.30
    w_volcomp: float = 0.15
    w_momentum: float = 0.15

    # Factor weights
    fw_momentum: float = 0.30
    fw_volume_surge: float = 0.15
    fw_realized_vol: float = 0.15
    fw_proximity: float = 0.15
    fw_rolling_sharpe: float = 0.25

    # Portfolio
    entry_threshold: float = 0.65
    exit_score_threshold: float = 0.40
    regime_entry_min: float = 0.45
    regime_exit_min: float = 0.15
    max_hold_hours: int = 336
    trailing_stop_pct: float = 0.15
    rebalance_every_hours: int = 24
    max_positions: int = 25
    max_weight: float = 0.10
    ema_span: int = 72

    # Data
    lookback_hours: int = 5000  # enough for 200d SMA


class MedallionSignalService:
    """Stateful signal service that computes target weights each cycle.

    Usage:
        config = SignalConfig(db_path="/path/to/market.duckdb")
        svc = MedallionSignalService(config)
        output = svc.run_cycle()
        # output.target_weights is {symbol: weight} for the execution engine
    """

    def __init__(self, config: SignalConfig) -> None:
        self.config = config
        self.state = PortfolioState.load(Path(config.state_path))

    def run_cycle(self) -> SignalOutput:
        """Execute one signal cycle. Call this every hour."""
        now = datetime.now(timezone.utc)
        cycle_id = str(uuid.uuid4())[:8]
        cfg = self.config
        actions: list[TradeAction] = []

        logger.info("[cycle %s] Starting at %s", cycle_id, now.isoformat())

        # ── 1. Load data ──────────────────────────────────────────
        panel = self._load_data()
        if panel.empty:
            logger.error("[cycle %s] No data loaded", cycle_id)
            return self._empty_output(now, cycle_id, stale=True)

        freshness_hours = (
            now - panel["ts"].max().to_pydatetime().replace(tzinfo=timezone.utc)
        ).total_seconds() / 3600
        stale = freshness_hours > 3.0
        if stale:
            logger.warning(
                "[cycle %s] Data is %.1fh stale (threshold: 3h)",
                cycle_id, freshness_hours,
            )

        # ── 2. Build matrices ─────────────────────────────────────
        df = panel.sort_values(["symbol", "ts"])
        df["ret"] = df.groupby("symbol")["close"].pct_change()
        symbols = sorted(df["symbol"].unique())

        returns_wide = df.pivot(index="ts", columns="symbol", values="ret").sort_index().fillna(0)
        close_wide = df.pivot(index="ts", columns="symbol", values="close").sort_index()
        high_wide = df.pivot(index="ts", columns="symbol", values="high").sort_index()
        volume_wide = df.pivot(index="ts", columns="symbol", values="volume").sort_index().fillna(0)

        # ── 3. Regime ─────────────────────────────────────────────
        regime_score = self._compute_regime(panel, returns_wide)
        current_regime = float(regime_score.iloc[-1]) if len(regime_score) > 0 else 0.0

        # ── 4. Factors ────────────────────────────────────────────
        composite = self._compute_factors(close_wide, volume_wide, high_wide)
        score_smooth = composite.ewm(span=cfg.ema_span, min_periods=24).mean()
        latest_scores = score_smooth.iloc[-1] if len(score_smooth) > 0 else pd.Series(dtype=float)

        # ── 5. Update holdings + check exits ──────────────────────
        latest_ret = returns_wide.iloc[-1] if len(returns_wide) > 0 else pd.Series(0.0, index=symbols)

        for sym in list(self.state.holdings.keys()):
            h = self.state.holdings[sym]
            r = float(latest_ret.get(sym, 0.0))
            h["hours_held"] = h.get("hours_held", 0) + 1
            h["cum_ret"] = (1 + h.get("cum_ret", 0)) * (1 + r) - 1
            h["peak_cum"] = max(h.get("peak_cum", 0), h["cum_ret"])

            exit_action = None

            # Exit: regime collapse
            if current_regime < cfg.regime_exit_min:
                exit_action = "exit_regime"

            # Exit: trailing stop
            elif h["peak_cum"] > 0.02 and (h["peak_cum"] - h["cum_ret"]) > cfg.trailing_stop_pct:
                exit_action = "exit_stop"

            # Exit: max hold
            elif h["hours_held"] >= cfg.max_hold_hours:
                exit_action = "exit_maxhold"

            # Exit: factor degradation (at rebalance points)
            elif (self.state.cycle_count % cfg.rebalance_every_hours == 0
                  and sym in latest_scores.index
                  and latest_scores[sym] < cfg.exit_score_threshold):
                exit_action = "exit_factor"

            if exit_action:
                actions.append(TradeAction(
                    symbol=sym, action=exit_action, target_weight=0.0,
                    hours_held=h["hours_held"], cum_ret=h["cum_ret"],
                ))
                del self.state.holdings[sym]

        # ── 6. Check entries (at rebalance points) ────────────────
        is_rebalance = self.state.cycle_count % cfg.rebalance_every_hours == 0

        if is_rebalance and current_regime >= cfg.regime_entry_min:
            candidates = latest_scores[latest_scores > cfg.entry_threshold].dropna()
            candidates = candidates.drop(
                labels=[s for s in self.state.holdings if s in candidates.index],
                errors="ignore",
            )
            n_open = cfg.max_positions - len(self.state.holdings)
            if n_open > 0 and len(candidates) > 0:
                entries = candidates.sort_values(ascending=False).head(n_open)
                for sym in entries.index:
                    self.state.holdings[sym] = dict(
                        symbol=sym,
                        entry_ts=now.isoformat(),
                        entry_score=float(entries[sym]),
                        hours_held=0,
                        cum_ret=0.0,
                        peak_cum=0.0,
                    )
                    actions.append(TradeAction(
                        symbol=sym, action="entry",
                        target_weight=0.0,  # filled below
                        score=float(entries[sym]),
                    ))

        # ── 7. Compute target weights ─────────────────────────────
        target_weights = self._compute_weights(
            symbols, current_regime, returns_wide,
        )

        # Update entry actions with actual weights
        for a in actions:
            if a.action == "entry":
                a.target_weight = target_weights.get(a.symbol, 0.0)

        # ── 8. Publish ────────────────────────────────────────────
        self.state.last_cycle_ts = now.isoformat()
        self.state.cycle_count += 1
        if is_rebalance:
            self.state.last_rebalance_hour = self.state.cycle_count

        output = SignalOutput(
            ts=now.isoformat(),
            cycle_id=cycle_id,
            target_weights=target_weights,
            regime_score=current_regime,
            actions=[asdict(a) for a in actions],
            diagnostics=dict(
                n_holdings=len(self.state.holdings),
                gross_exposure=sum(target_weights.values()),
                n_eligible=int((latest_scores > cfg.entry_threshold).sum())
                    if len(latest_scores) > 0 else 0,
                data_freshness_hours=round(freshness_hours, 2),
                cycle_count=self.state.cycle_count,
                is_rebalance=is_rebalance,
            ),
            stale=stale,
        )

        self._publish(output)
        self.state.save(Path(self.config.state_path))

        logger.info(
            "[cycle %s] Done: %d holdings, %.1f%% exposure, regime=%.2f, %d actions",
            cycle_id, len(self.state.holdings),
            sum(target_weights.values()) * 100,
            current_regime, len(actions),
        )
        return output

    # ── Internal Methods ────────────────────────────────────────────

    def _load_data(self) -> pd.DataFrame:
        """Load recent hourly bars from DuckDB."""
        cfg = self.config
        con = duckdb.connect(cfg.db_path, read_only=True)
        try:
            top_syms = con.execute(f"""
                WITH daily_dv AS (
                    SELECT symbol,
                           date_trunc('day', ts) AS day,
                           SUM(close * volume) AS dv
                    FROM bars_1h
                    WHERE ts >= current_timestamp - INTERVAL '{cfg.lookback_hours} hours'
                      AND close > 0 AND volume > 0
                    GROUP BY symbol, date_trunc('day', ts)
                )
                SELECT symbol, MEDIAN(dv) AS med_dv
                FROM daily_dv
                GROUP BY symbol
                HAVING MEDIAN(dv) >= {cfg.min_adv_usd}
                ORDER BY med_dv DESC
                LIMIT {cfg.max_symbols}
            """).fetchall()
            sym_list = [r[0] for r in top_syms]

            if not sym_list:
                return pd.DataFrame()

            placeholders = ", ".join(["?"] * len(sym_list))
            df = con.execute(f"""
                SELECT symbol, ts, open, high, low, close, volume
                FROM bars_1h
                WHERE ts >= current_timestamp - INTERVAL '{cfg.lookback_hours} hours'
                  AND symbol IN ({placeholders})
                  AND open > 0 AND close > 0 AND high >= low
                ORDER BY symbol, ts
            """, sym_list).fetch_df()
        finally:
            con.close()

        df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_localize(None)

        # Universe filter: require min history
        counts = df.groupby("symbol")["ts"].transform("count")
        df = df[counts >= cfg.min_history_hours].copy()

        logger.info(
            "Loaded %d rows, %d symbols",
            len(df), df["symbol"].nunique(),
        )
        return df

    def _compute_regime(
        self, panel: pd.DataFrame, returns_wide: pd.DataFrame,
    ) -> pd.Series:
        """Compute ensemble regime score."""
        cfg = self.config

        if "BTC-USD" not in panel["symbol"].values:
            return pd.Series(1.0, index=returns_wide.index)

        btc_h = panel[panel["symbol"] == "BTC-USD"].set_index("ts")["close"].sort_index()
        btc_d = btc_h.resample("D").last().dropna()

        # BTC trend
        sma_f = btc_d.rolling(50, min_periods=50).mean()
        sma_s = btc_d.rolling(200, min_periods=200).mean()
        trend = pd.Series(0.0, index=btc_d.index)
        trend[btc_d > sma_f] = 0.6
        trend[(btc_d > sma_f) & (sma_f > sma_s)] = 1.0
        trend_h = trend.reindex(returns_wide.index, method="ffill")

        # Breadth
        cum = returns_wide.rolling(168, min_periods=84).sum()
        breadth = (cum > 0).mean(axis=1)

        # Vol compression
        ret_btc = btc_h.pct_change()
        vol_s = ret_btc.rolling(24, min_periods=24).std()
        vol_l = ret_btc.rolling(168, min_periods=84).std()
        ratio = vol_l / vol_s.clip(lower=1e-8)
        vol_comp = (ratio - 0.5).clip(0, 2) / 2.0
        vol_comp = vol_comp.reindex(returns_wide.index, method="ffill")

        # BTC momentum
        ret_168 = btc_h.pct_change(168)
        momentum = (ret_168.clip(-0.5, 0.5) / 0.5 + 1) / 2
        momentum = momentum.reindex(returns_wide.index, method="ffill")

        score = (
            cfg.w_trend * trend_h.fillna(0)
            + cfg.w_breadth * breadth.fillna(0)
            + cfg.w_volcomp * vol_comp.fillna(0.5)
            + cfg.w_momentum * momentum.fillna(0.5)
        ).clip(0, 1)

        return score

    def _compute_factors(
        self,
        close_wide: pd.DataFrame,
        volume_wide: pd.DataFrame,
        high_wide: pd.DataFrame,
    ) -> pd.DataFrame:
        """Compute composite factor score."""
        cfg = self.config
        ret = close_wide.pct_change()

        momentum = np.log(close_wide / close_wide.shift(168))
        vol_24h = volume_wide.rolling(24, min_periods=12).sum()
        vol_7d_avg = volume_wide.rolling(168, min_periods=84).mean() * 24
        volume_surge = (vol_24h / vol_7d_avg.clip(lower=1)).clip(0, 5)
        realized_vol = ret.rolling(168, min_periods=84).std() * np.sqrt(ANN_FACTOR)
        rolling_high = high_wide.rolling(168, min_periods=24).max()
        proximity = 1 + (close_wide - rolling_high) / rolling_high.clip(lower=1e-8)
        roll_mean = ret.rolling(168, min_periods=84).mean()
        roll_std = ret.rolling(168, min_periods=84).std()
        rolling_sharpe = roll_mean / roll_std.clip(lower=1e-8)

        factors = {
            "momentum": momentum,
            "volume_surge": volume_surge,
            "realized_vol": realized_vol,
            "proximity_to_high": proximity,
            "rolling_sharpe": rolling_sharpe,
        }
        weights = {
            "momentum": cfg.fw_momentum,
            "volume_surge": cfg.fw_volume_surge,
            "realized_vol": cfg.fw_realized_vol,
            "proximity_to_high": cfg.fw_proximity,
            "rolling_sharpe": cfg.fw_rolling_sharpe,
        }

        ref = next(iter(factors.values()))
        composite = pd.DataFrame(0.0, index=ref.index, columns=ref.columns)
        for name, w in weights.items():
            ranked = factors[name].rank(axis=1, pct=True)
            composite += w * ranked.fillna(0.5)
        return composite

    def _compute_weights(
        self,
        symbols: list[str],
        regime_score: float,
        returns_wide: pd.DataFrame,
    ) -> dict[str, float]:
        """Compute target weights from current holdings."""
        cfg = self.config
        if not self.state.holdings:
            return {s: 0.0 for s in symbols}

        held = list(self.state.holdings.keys())
        vol = returns_wide.iloc[-168:].std() * np.sqrt(ANN_FACTOR) if len(returns_wide) >= 168 else None

        if vol is not None:
            v = vol.reindex(held).fillna(vol.median()).clip(lower=0.1)
            raw = 1.0 / v
        else:
            raw = pd.Series(1.0, index=held)

        regime_scale = max(regime_score, 0.20) if regime_score >= cfg.regime_exit_min else 0.0
        raw = raw * regime_scale

        total = raw.sum()
        if total > 0:
            raw = raw / total

        raw = raw.clip(upper=cfg.max_weight)
        total = raw.sum()
        if total > 0:
            raw = raw / total

        target = {s: 0.0 for s in symbols}
        for sym in held:
            if sym in raw.index:
                target[sym] = round(float(raw[sym]), 6)
        return target

    def _publish(self, output: SignalOutput) -> None:
        """Write signal output to JSON file and optionally DuckDB."""
        out_path = Path(self.config.output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(asdict(output), indent=2, default=str))
        logger.info("Published to %s", out_path)

        # Also write to DuckDB for queryable history
        self._publish_to_db(output)

    def _publish_to_db(self, output: SignalOutput) -> None:
        """Append signal to DuckDB live_signals table."""
        try:
            con = duckdb.connect(self.config.db_path)
            con.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.config.output_db_table} (
                    ts TIMESTAMP,
                    cycle_id VARCHAR,
                    target_weights JSON,
                    regime_score DOUBLE,
                    n_holdings INTEGER,
                    gross_exposure DOUBLE,
                    stale BOOLEAN,
                    raw_json JSON
                )
            """)
            con.execute(
                f"INSERT INTO {self.config.output_db_table} VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    output.ts,
                    output.cycle_id,
                    json.dumps(output.target_weights),
                    output.regime_score,
                    output.diagnostics.get("n_holdings", 0),
                    output.diagnostics.get("gross_exposure", 0),
                    output.stale,
                    json.dumps(asdict(output), default=str),
                ],
            )
            con.close()
        except Exception as e:
            logger.warning("Failed to write to DuckDB: %s", e)

    def _empty_output(self, now: datetime, cycle_id: str, stale: bool) -> SignalOutput:
        return SignalOutput(
            ts=now.isoformat(),
            cycle_id=cycle_id,
            target_weights={},
            regime_score=0.0,
            actions=[],
            diagnostics=dict(n_holdings=0, gross_exposure=0.0, error="no_data"),
            stale=stale,
        )
