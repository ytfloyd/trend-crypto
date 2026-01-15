from src.common.config import StrategyConfig
from src.strategy.ma_crossover_long_only import MACrossoverLongOnlyStrategy


def test_adx_filter_default_false_in_config():
    cfg = StrategyConfig(mode="ma_crossover_long_only", fast=5, slow=40)
    assert cfg.enable_adx_filter is False


def test_adx_filter_explicit_true_in_config():
    cfg = StrategyConfig(mode="ma_crossover_long_only", fast=5, slow=40, enable_adx_filter=True)
    assert cfg.enable_adx_filter is True


def test_adx_filter_default_false_in_strategy():
    strat = MACrossoverLongOnlyStrategy(fast=5, slow=40)
    assert strat.enable_adx_filter is False
