from __future__ import annotations

import numpy as np
import pandas as pd

from src.metrics import performance_summary, sharpe_ratio
from src.simulation import simulate_portfolio


def test_sharpe_uses_cash_proxy_as_hurdle() -> None:
    dates = pd.bdate_range("2020-01-01", periods=20)
    returns = pd.Series(np.linspace(-0.002, 0.004, len(dates)), index=dates)
    cash = pd.Series(0.0002, index=dates)
    expected = (returns - cash).mean() / (returns - cash).std(ddof=0) * np.sqrt(252.0)
    assert np.isclose(sharpe_ratio(returns, risk_free_returns=cash), expected)


def test_effective_assets_is_invariant_to_leverage() -> None:
    dates = pd.bdate_range("2020-01-01", periods=10)
    asset_returns = pd.DataFrame(0.0, index=dates, columns=["A", "B"])
    cash_returns = pd.Series(0.0, index=dates)
    targets = {
        "unlevered": pd.DataFrame({"A": [0.5], "B": [0.5]}, index=[dates[0]]),
        "levered": pd.DataFrame({"A": [0.75], "B": [0.75]}, index=[dates[0]]),
    }
    simulations = {
        name: simulate_portfolio(asset_returns, target, cash_returns, transaction_cost_bps=0.0)
        for name, target in targets.items()
    }
    summary = performance_summary(simulations, cash_returns=cash_returns)
    assert np.isclose(summary.loc["unlevered", "effective_assets_by_weight"], 2.0)
    assert np.isclose(summary.loc["levered", "effective_assets_by_weight"], 2.0)
