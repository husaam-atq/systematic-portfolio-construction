from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest import generate_target_weights, volatility_target_weights
from src.config import ResearchConfig
from src.data import calculate_returns
from src.signals import dual_momentum, trend_eligibility
from src.simulation import simulate_portfolio


def test_estimation_window_excludes_future_returns(synthetic_prices: pd.DataFrame) -> None:
    config = ResearchConfig(estimation_window=126)
    returns = calculate_returns(synthetic_prices)
    targets, diagnostics, _ = generate_target_weights(
        returns,
        synthetic_prices,
        config,
        methods=["Minimum Variance"],
    )
    decision = targets["Minimum Variance"].index[3]
    changed_prices = synthetic_prices.copy()
    changed_prices.loc[changed_prices.index > decision, "SPY"] *= 3.0
    changed_returns = calculate_returns(changed_prices)
    changed_targets, _, _ = generate_target_weights(
        changed_returns,
        changed_prices,
        config,
        methods=["Minimum Variance"],
    )
    pd.testing.assert_series_equal(
        targets["Minimum Variance"].loc[decision],
        changed_targets["Minimum Variance"].loc[decision],
    )
    assert (pd.to_datetime(diagnostics["input_end"]) <= pd.to_datetime(diagnostics["date"])).all()


def test_trend_and_momentum_signals_are_causal(synthetic_prices: pd.DataFrame) -> None:
    date = synthetic_prices.index[400]
    trend = trend_eligibility(synthetic_prices)
    momentum = dual_momentum(synthetic_prices)
    changed = synthetic_prices.copy()
    changed.loc[changed.index > date] *= 5.0
    pd.testing.assert_series_equal(trend.loc[date], trend_eligibility(changed).loc[date])
    pd.testing.assert_series_equal(momentum.loc[date], dual_momentum(changed).loc[date])


def test_volatility_target_scale_is_lagged(synthetic_prices: pd.DataFrame) -> None:
    returns = calculate_returns(synthetic_prices)
    first = returns.index[0]
    target = pd.DataFrame(
        [np.repeat(1.0 / returns.shape[1], returns.shape[1])],
        index=[first],
        columns=returns.columns,
    )
    base = simulate_portfolio(returns, target, returns["SHY"], transaction_cost_bps=0.0)
    _, scale = volatility_target_weights(base, returns, 0.10, 63, 1.5)
    test_date = scale.dropna().index[5]
    changed_performance = base.performance.copy()
    changed_performance.loc[test_date, "gross_return"] = 0.50
    changed_base = type(base)(
        changed_performance,
        base.pre_trade_weights,
        base.post_trade_weights,
        base.asset_return_contributions,
        base.target_weights,
    )
    _, changed_scale = volatility_target_weights(changed_base, returns, 0.10, 63, 1.5)
    assert np.isclose(scale.loc[test_date], changed_scale.loc[test_date])
    assert not np.isclose(scale.shift(-1).loc[test_date], changed_scale.shift(-1).loc[test_date])
