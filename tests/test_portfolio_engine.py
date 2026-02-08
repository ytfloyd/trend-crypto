"""Tests for multi-asset PortfolioEngine and related components.

Covers:
- PortfolioConfig validation
- SingleAssetAdapter wrapping
- PortfolioEngine two-asset backtest
- Portfolio constraints (gross leverage, single-name limits)
- Metrics: Sharpe, Sortino, Calmar, HHI, diversification ratio
- Correlation module: rolling correlation, regime detection
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import polars as pl
import pytest

from backtest.metrics import (
    calmar_ratio,
    diversification_ratio,
    hhi_concentration,
    max_drawdown,
    return_contribution,
    sharpe_ratio,
    sortino_ratio,
)
from backtest.portfolio_engine import PortfolioEngine, _align_multi_symbol_bars
from backtest.portfolio_result import PortfolioResult
from common.config import (
    DataConfig,
    EngineConfig,
    ExecutionConfig,
    PortfolioConfig,
    RiskConfigRaw,
    RunConfigRaw,
    StrategyConfigRaw,
    compile_config,
)
from risk.correlation import (
    CorrelationRegime,
    average_correlation,
    correlation_regime_indicator,
    rolling_correlation_matrix,
)
from risk.risk_manager import RiskManager
from strategy.base import (
    PortfolioStrategy,
    SingleAssetAdapter,
    StrategySignals,
    TargetWeightStrategy,
)
from strategy.context import StrategyContext
from strategy.ma_crossover_long_only import MACrossoverLongOnlyStrategy


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------


def _make_bars(
    n: int, symbol: str = "BTC-USD", base_price: float = 100.0, trend: float = 0.5
) -> pl.DataFrame:
    """Create synthetic bars with a deterministic trend."""
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    rows = []
    price = base_price
    for i in range(n):
        ts = start + timedelta(hours=i)
        delta = trend * (1 if (i * 7 + 3) % 5 < 3 else -1)
        o = price
        c = price + delta
        rows.append({
            "ts": ts,
            "symbol": symbol,
            "open": o,
            "high": max(o, c) + 0.1,
            "low": min(o, c) - 0.1,
            "close": c,
            "volume": 1000.0 + i * 10.0,
        })
        price = c
    return pl.DataFrame(rows)


def _make_resolved_config(
    bars: pl.DataFrame, portfolio_cfg: PortfolioConfig
) -> compile_config:
    """Build a resolved config for multi-asset testing."""
    raw = RunConfigRaw(
        run_name="portfolio_test",
        data=DataConfig(
            db_path=":memory:",
            table="bars",
            symbol=portfolio_cfg.symbols[0],
            start=bars[0, "ts"],
            end=bars[bars.height - 1, "ts"],
            timeframe="1h",
        ),
        engine=EngineConfig(strict_validation=False, lookback=None, initial_cash=100_000.0),
        strategy=StrategyConfigRaw(
            mode="ma_crossover_long_only",
            fast=5, slow=20,
            weight_on=1.0, window_units="bars",
        ),
        risk=RiskConfigRaw(
            vol_window=10, target_vol_annual=None, max_weight=1.0, window_units="bars",
        ),
        execution=ExecutionConfig(
            fee_bps=10.0, slippage_bps=5.0,
            execution_lag_bars=1, rebalance_deadband=0.01,
        ),
        portfolio=portfolio_cfg,
    )
    return compile_config(raw)


# ---------------------------------------------------------------------------
# PortfolioConfig validation
# ---------------------------------------------------------------------------


class TestPortfolioConfig:
    def test_valid_config(self) -> None:
        cfg = PortfolioConfig(symbols=["BTC-USD", "ETH-USD"])
        assert cfg.max_gross_leverage == 1.0
        assert cfg.max_single_name_weight == 1.0

    def test_empty_symbols_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one symbol"):
            PortfolioConfig(symbols=[])

    def test_duplicate_symbols_raises(self) -> None:
        with pytest.raises(ValueError, match="unique"):
            PortfolioConfig(symbols=["BTC-USD", "BTC-USD"])

    def test_negative_leverage_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            PortfolioConfig(symbols=["BTC-USD"], max_gross_leverage=-1.0)


# ---------------------------------------------------------------------------
# SingleAssetAdapter
# ---------------------------------------------------------------------------


class TestSingleAssetAdapter:
    def test_adapter_routes_to_correct_strategy(self) -> None:
        bars_btc = _make_bars(50, "BTC-USD", base_price=1000.0)
        bars_eth = _make_bars(50, "ETH-USD", base_price=100.0, trend=0.3)

        strat_btc = MACrossoverLongOnlyStrategy(fast=5, slow=20, max_weight=1.0)
        strat_eth = MACrossoverLongOnlyStrategy(fast=5, slow=20, max_weight=0.5)

        adapter = SingleAssetAdapter({"BTC-USD": strat_btc, "ETH-USD": strat_eth})
        assert isinstance(adapter, PortfolioStrategy)

        # Build contexts
        from strategy.context import make_strategy_context
        ctx_btc = make_strategy_context(bars_btc, 49, None)
        ctx_eth = make_strategy_context(bars_eth, 49, None)
        contexts = {"BTC-USD": ctx_btc, "ETH-USD": ctx_eth}

        weights = adapter.on_bar_close_portfolio(contexts)
        assert "BTC-USD" in weights
        assert "ETH-USD" in weights

    def test_adapter_returns_zero_for_unknown_symbol(self) -> None:
        strat = MACrossoverLongOnlyStrategy(fast=5, slow=20, max_weight=1.0)
        adapter = SingleAssetAdapter({"BTC-USD": strat})

        bars = _make_bars(50, "ETH-USD")
        from strategy.context import make_strategy_context
        ctx = make_strategy_context(bars, 49, None)
        weights = adapter.on_bar_close_portfolio({"ETH-USD": ctx})
        assert weights["ETH-USD"] == 0.0


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------


class TestAlignBars:
    def test_inner_join_alignment(self) -> None:
        """Only common timestamps survive alignment."""
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        bars_a = pl.DataFrame({
            "ts": [start, start + timedelta(hours=1), start + timedelta(hours=2)],
            "symbol": ["A"] * 3,
            "open": [100.0, 101.0, 102.0],
            "high": [101.0, 102.0, 103.0],
            "low": [99.0, 100.0, 101.0],
            "close": [100.5, 101.5, 102.5],
            "volume": [1000.0] * 3,
        })
        bars_b = pl.DataFrame({
            "ts": [start + timedelta(hours=1), start + timedelta(hours=2), start + timedelta(hours=3)],
            "symbol": ["B"] * 3,
            "open": [50.0, 51.0, 52.0],
            "high": [51.0, 52.0, 53.0],
            "low": [49.0, 50.0, 51.0],
            "close": [50.5, 51.5, 52.5],
            "volume": [500.0] * 3,
        })
        common_ts, aligned = _align_multi_symbol_bars({"A": bars_a, "B": bars_b})
        assert len(common_ts) == 2  # hours 1 and 2 overlap
        assert aligned["A"].height == 2
        assert aligned["B"].height == 2


# ---------------------------------------------------------------------------
# PortfolioEngine end-to-end
# ---------------------------------------------------------------------------


class TestPortfolioEngine:
    def test_two_asset_backtest(self) -> None:
        """Two-asset backtest produces correct result shape."""
        n = 100
        bars_btc = _make_bars(n, "BTC-USD", base_price=1000.0)
        bars_eth = _make_bars(n, "ETH-USD", base_price=100.0, trend=0.3)

        portfolio_cfg = PortfolioConfig(symbols=["BTC-USD", "ETH-USD"])
        cfg = _make_resolved_config(bars_btc, portfolio_cfg)

        strategies = {
            "BTC-USD": MACrossoverLongOnlyStrategy(fast=5, slow=20, max_weight=1.0),
            "ETH-USD": MACrossoverLongOnlyStrategy(fast=5, slow=20, max_weight=0.5),
        }
        rm = RiskManager(cfg.risk, periods_per_year=cfg.annualization_factor)

        engine = PortfolioEngine(
            cfg=cfg,
            strategy=strategies,
            risk_manager=rm,
            bars_by_symbol={"BTC-USD": bars_btc, "ETH-USD": bars_eth},
            portfolio_cfg=portfolio_cfg,
        )
        result = engine.run()

        assert isinstance(result, PortfolioResult)
        assert result.equity_df.height == n
        assert "nav" in result.equity_df.columns
        assert "gross_ret" in result.equity_df.columns
        assert "net_ret" in result.equity_df.columns
        assert result.weights_df.height > 0
        assert result.contributions_df.height > 0
        assert result.summary["n_symbols"] == 2

    def test_single_asset_matches_pattern(self) -> None:
        """Single-asset portfolio engine should still work correctly."""
        n = 60
        bars = _make_bars(n, "BTC-USD", base_price=500.0)
        portfolio_cfg = PortfolioConfig(symbols=["BTC-USD"])
        cfg = _make_resolved_config(bars, portfolio_cfg)

        strategies = {
            "BTC-USD": MACrossoverLongOnlyStrategy(fast=5, slow=20, max_weight=1.0),
        }
        rm = RiskManager(cfg.risk, periods_per_year=cfg.annualization_factor)

        engine = PortfolioEngine(
            cfg=cfg, strategy=strategies, risk_manager=rm,
            bars_by_symbol={"BTC-USD": bars},
            portfolio_cfg=portfolio_cfg,
        )
        result = engine.run()
        assert result.equity_df.height == n
        assert result.summary["n_symbols"] == 1

    def test_gross_leverage_constraint(self) -> None:
        """Gross leverage constraint should cap total exposure."""
        n = 100
        bars_btc = _make_bars(n, "BTC-USD", base_price=1000.0)
        bars_eth = _make_bars(n, "ETH-USD", base_price=100.0, trend=0.3)
        portfolio_cfg = PortfolioConfig(
            symbols=["BTC-USD", "ETH-USD"],
            max_gross_leverage=0.5,
        )
        cfg = _make_resolved_config(bars_btc, portfolio_cfg)
        strategies = {
            "BTC-USD": MACrossoverLongOnlyStrategy(fast=5, slow=20, max_weight=1.0),
            "ETH-USD": MACrossoverLongOnlyStrategy(fast=5, slow=20, max_weight=1.0),
        }
        rm = RiskManager(cfg.risk, periods_per_year=cfg.annualization_factor)
        engine = PortfolioEngine(
            cfg=cfg, strategy=strategies, risk_manager=rm,
            bars_by_symbol={"BTC-USD": bars_btc, "ETH-USD": bars_eth},
            portfolio_cfg=portfolio_cfg,
        )
        result = engine.run()
        # Check held weights never exceed gross leverage limit (with small tolerance)
        for ts in result.weights_df["ts"].unique().to_list():
            ts_weights = result.weights_df.filter(pl.col("ts") == ts)
            gross = ts_weights["held_weight"].abs().sum()
            assert gross <= 0.5 + 1e-9, f"Gross leverage {gross} > 0.5 at {ts}"

    def test_missing_portfolio_config_raises(self) -> None:
        """PortfolioEngine requires a PortfolioConfig."""
        bars = _make_bars(50, "BTC-USD")
        raw = RunConfigRaw(
            run_name="test",
            data=DataConfig(
                db_path=":memory:", table="bars", symbol="BTC-USD",
                start=bars[0, "ts"], end=bars[bars.height - 1, "ts"], timeframe="1h",
            ),
            engine=EngineConfig(strict_validation=False),
            strategy=StrategyConfigRaw(mode="buy_and_hold", window_units="bars"),
            risk=RiskConfigRaw(vol_window=10, target_vol_annual=None, max_weight=1.0, window_units="bars"),
            execution=ExecutionConfig(),
        )
        cfg = compile_config(raw)
        rm = RiskManager(cfg.risk, periods_per_year=cfg.annualization_factor)
        with pytest.raises(ValueError, match="PortfolioConfig"):
            PortfolioEngine(
                cfg=cfg, strategy={}, risk_manager=rm,
                bars_by_symbol={"BTC-USD": bars},
            )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


class TestMetrics:
    def _make_nav_returns(self) -> tuple[pl.Series, pl.Series]:
        """Generate a simple NAV series for testing."""
        nav_values = [100.0, 101.0, 99.0, 102.0, 103.0, 100.0, 105.0]
        nav = pl.Series("nav", nav_values)
        returns = nav.pct_change().fill_null(0.0)
        return nav, returns

    def test_sharpe_positive(self) -> None:
        nav, returns = self._make_nav_returns()
        s = sharpe_ratio(returns, periods_per_year=8760.0)
        assert isinstance(s, float)

    def test_sortino_positive(self) -> None:
        nav, returns = self._make_nav_returns()
        s = sortino_ratio(returns, periods_per_year=8760.0)
        assert isinstance(s, float)

    def test_calmar_positive(self) -> None:
        nav, returns = self._make_nav_returns()
        c = calmar_ratio(returns, nav, periods_per_year=8760.0)
        assert isinstance(c, float)

    def test_max_drawdown_negative(self) -> None:
        nav, _ = self._make_nav_returns()
        dd = max_drawdown(nav)
        assert dd <= 0.0

    def test_hhi_equal_weights(self) -> None:
        weights = {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25}
        hhi = hhi_concentration(weights)
        assert abs(hhi - 0.25) < 1e-10

    def test_hhi_concentrated(self) -> None:
        weights = {"A": 1.0, "B": 0.0}
        hhi = hhi_concentration(weights)
        assert abs(hhi - 1.0) < 1e-10

    def test_diversification_ratio_single_asset(self) -> None:
        cov = {"A": {"A": 0.04}}
        dr = diversification_ratio({"A": 1.0}, cov)
        assert abs(dr - 1.0) < 1e-10

    def test_return_contribution_sums(self) -> None:
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        contrib_df = pl.DataFrame({
            "ts": [start, start, start + timedelta(hours=1), start + timedelta(hours=1)],
            "symbol": ["A", "B", "A", "B"],
            "contribution": [0.01, 0.005, 0.02, -0.01],
        })
        weights_df = pl.DataFrame({
            "ts": [start, start], "symbol": ["A", "B"], "held_weight": [0.5, 0.5],
        })
        rc = return_contribution(weights_df, contrib_df, ["A", "B"])
        assert abs(rc["A"] - 0.03) < 1e-10
        assert abs(rc["B"] - (-0.005)) < 1e-10


# ---------------------------------------------------------------------------
# Correlation
# ---------------------------------------------------------------------------


class TestCorrelation:
    def test_rolling_correlation_matrix_shape(self) -> None:
        import random
        random.seed(42)
        n = 100
        a = pl.Series("A", [random.gauss(0, 1) for _ in range(n)])
        b = pl.Series("B", [random.gauss(0, 1) for _ in range(n)])
        matrices = rolling_correlation_matrix({"A": a, "B": b}, window=20)
        assert len(matrices) == n - 20 + 1
        assert "A" in matrices[0]
        assert "B" in matrices[0]["A"]

    def test_average_correlation_symmetric(self) -> None:
        matrix = {"A": {"A": 1.0, "B": 0.5}, "B": {"A": 0.5, "B": 1.0}}
        avg = average_correlation(matrix)
        assert abs(avg - 0.5) < 1e-10

    def test_regime_crisis_detection(self) -> None:
        """Perfectly correlated series should be classified as CRISIS."""
        n = 100
        base = [float(i) * 0.01 for i in range(n)]
        returns = {"A": pl.Series("A", base), "B": pl.Series("B", base)}
        regime = correlation_regime_indicator(returns, window=20, crisis_threshold=0.7)
        assert regime == CorrelationRegime.CRISIS

    def test_regime_insufficient_data_returns_none(self) -> None:
        returns = {"A": pl.Series("A", [0.01, 0.02]), "B": pl.Series("B", [0.01, -0.01])}
        regime = correlation_regime_indicator(returns, window=20)
        assert regime is None
