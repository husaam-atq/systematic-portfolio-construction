from __future__ import annotations

import numpy as np
import pandas as pd

from src.allocation import (
    covariance_matrix,
    hierarchical_risk_parity,
    maximum_sharpe,
    minimum_variance,
    percentage_risk_contributions,
    risk_parity,
)
from src.data import calculate_returns


def test_covariance_shrinkage_moves_off_diagonal_toward_zero(synthetic_prices: pd.DataFrame) -> None:
    returns = calculate_returns(synthetic_prices).iloc[-252:]
    sample = covariance_matrix(returns, shrinkage=0.0)
    shrunk = covariance_matrix(returns, shrinkage=0.75)
    mask = ~np.eye(len(sample), dtype=bool)
    assert np.abs(shrunk.to_numpy()[mask]).mean() < np.abs(sample.to_numpy()[mask]).mean()
    assert np.allclose(np.diag(shrunk), np.diag(sample))


def test_optimizers_respect_long_only_sum_and_cap(synthetic_prices: pd.DataFrame) -> None:
    returns = calculate_returns(synthetic_prices).iloc[-252:]
    for result in [minimum_variance(returns), maximum_sharpe(returns), hierarchical_risk_parity(returns)]:
        assert result.diagnostics["solver_success"]
        assert np.isclose(result.weights.sum(), 1.0)
        assert (result.weights >= -1e-12).all()
        assert (result.weights <= 0.40 + 1e-8).all()


def test_risk_parity_equalizes_ex_ante_risk(synthetic_prices: pd.DataFrame) -> None:
    returns = calculate_returns(synthetic_prices).iloc[-252:]
    result = risk_parity(returns)
    contributions = percentage_risk_contributions(result.weights, result.covariance)
    assert np.isclose(contributions.sum(), 1.0)
    assert (contributions - 1.0 / len(contributions)).abs().max() < 0.02
