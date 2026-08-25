from __future__ import annotations

import numpy as np
import pandas as pd

from src.simulation import common_start_targets, simulate_portfolio


def _returns() -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-01", periods=4)
    return pd.DataFrame({"A": [0.10, 0.00, 0.00, 0.00], "B": [0.00, 0.00, 0.00, 0.00]}, index=dates)


def test_target_does_not_earn_same_day_return() -> None:
    returns = _returns()
    targets = pd.DataFrame({"A": [1.0], "B": [0.0]}, index=[returns.index[0]])
    result = simulate_portfolio(returns, targets, pd.Series(0.0, index=returns.index), transaction_cost_bps=0.0)
    assert result.performance.loc[returns.index[0], "gross_return"] == 0.0


def test_warmup_exposure_is_not_reported_as_active() -> None:
    returns = _returns()
    targets = pd.DataFrame({"A": [1.0], "B": [0.0]}, index=[returns.index[2]])
    result = simulate_portfolio(returns, targets, pd.Series(0.0, index=returns.index), transaction_cost_bps=0.0)
    assert result.performance.loc[returns.index[:2], "gross_exposure"].isna().all()
    assert result.performance.loc[returns.index[2], "gross_exposure"] == 1.0


def test_common_start_targets_reset_inception_consistently() -> None:
    returns = _returns()
    cash = pd.Series(0.0, index=returns.index)
    recurring = pd.DataFrame(
        {"A": [0.5, 0.5], "B": [0.5, 0.5]},
        index=[returns.index[0], returns.index[2]],
    )
    late = recurring.iloc[[1]]
    simulations = {
        "early": simulate_portfolio(returns, recurring, cash, transaction_cost_bps=0.0),
        "late": simulate_portfolio(returns, late, cash, transaction_cost_bps=0.0),
    }
    restarted = common_start_targets(simulations, returns.columns, returns.index[2])
    pd.testing.assert_series_equal(
        restarted["early"].iloc[0],
        restarted["late"].iloc[0],
        check_names=False,
    )


def test_holdings_drift_and_turnover_reconcile() -> None:
    returns = _returns()
    targets = pd.DataFrame(
        {"A": [0.5, 0.5], "B": [0.5, 0.5]},
        index=[returns.index[0], returns.index[1]],
    )
    result = simulate_portfolio(returns, targets, pd.Series(0.0, index=returns.index), transaction_cost_bps=5.0)
    expected_a = 0.5
    assert np.isclose(result.post_trade_weights.loc[returns.index[0], "A"], expected_a)
    assert np.isclose(result.performance.loc[returns.index[0], "turnover"], 1.0)
    assert np.isclose(
        result.performance.loc[returns.index[0], "transaction_cost_rate"],
        result.performance.loc[returns.index[0], "turnover"] * 5.0 / 10000.0,
    )


def test_drift_without_rebalance() -> None:
    returns = _returns()
    targets = pd.DataFrame({"A": [0.5], "B": [0.5]}, index=[returns.index[0]])
    result = simulate_portfolio(returns, targets, pd.Series(0.0, index=returns.index), transaction_cost_bps=0.0)
    returns.loc[returns.index[1], "A"] = 0.10
    result = simulate_portfolio(returns, targets, pd.Series(0.0, index=returns.index), transaction_cost_bps=0.0)
    assert np.isclose(result.post_trade_weights.loc[returns.index[1], "A"], 0.55 / 1.05)


def test_cash_and_financing_accounting() -> None:
    returns = _returns() * 0.0
    cash = pd.Series(0.001, index=returns.index)
    underinvested = pd.DataFrame({"A": [0.5], "B": [0.0]}, index=[returns.index[0]])
    result = simulate_portfolio(returns, underinvested, cash, transaction_cost_bps=0.0)
    assert np.isclose(result.performance.loc[returns.index[1], "gross_return"], 0.5 * 0.001)

    levered = pd.DataFrame({"A": [1.2], "B": [0.0]}, index=[returns.index[0]])
    result = simulate_portfolio(
        returns,
        levered,
        cash,
        transaction_cost_bps=0.0,
        financing_spread_annual=0.005,
    )
    expected = -0.2 * (0.001 + 0.005 / 252.0)
    assert np.isclose(result.performance.loc[returns.index[1], "gross_return"], expected)
    assert np.isclose(result.post_trade_weights.loc[returns.index[0]].sum(), 1.0)
