from __future__ import annotations

import pandas as pd

from src.allocation import STRATEGY_NAMES, estimate_allocation_weights
from src.risk import component_risk_contribution


def month_end_trading_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Return the last available trading date in each calendar month."""
    dates = pd.Series(index=index, data=index)
    return pd.DatetimeIndex(dates.groupby([index.year, index.month]).tail(1).to_list())


def generate_weight_decisions(
    returns: pd.DataFrame,
    estimation_window: int = 252,
    max_weight: float = 0.40,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """
    Estimate monthly allocation weights using only trailing returns.

    For a rebalance date at position i, the estimation window is
    returns.iloc[i - estimation_window:i], which excludes the rebalance date
    return and all future returns. The resulting decision is applied by
    run_backtest starting on the next trading session.
    """
    rebalance_dates = month_end_trading_dates(returns.index)
    decisions = {
        strategy: pd.DataFrame(index=rebalance_dates, columns=returns.columns, dtype=float)
        for strategy in STRATEGY_NAMES
    }
    risk_records: list[dict[str, float | str | pd.Timestamp]] = []

    for rebalance_date in rebalance_dates:
        position = returns.index.get_loc(rebalance_date)
        if position < estimation_window:
            continue

        trailing_returns = returns.iloc[position - estimation_window:position].dropna(how="any")
        if len(trailing_returns) < estimation_window:
            continue

        estimated = estimate_allocation_weights(trailing_returns, max_weight=max_weight)
        covariance = trailing_returns.cov()

        for strategy, weights in estimated.items():
            decisions[strategy].loc[rebalance_date, weights.index] = weights
            contribution, portfolio_volatility = component_risk_contribution(weights, covariance)
            for asset, value in contribution.items():
                risk_records.append(
                    {
                        "date": rebalance_date,
                        "strategy": strategy,
                        "asset": asset,
                        "component_risk_contribution": value,
                        "portfolio_volatility": portfolio_volatility,
                    }
                )

    risk_contributions = pd.DataFrame(risk_records)
    return decisions, risk_contributions


def run_backtest(
    returns: pd.DataFrame,
    decision_weights: pd.DataFrame,
    tc_bps: float = 5.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply rebalance decisions after they are calculated and subtract costs.

    decision_weights are timestamped on rebalance dates. The one-day shift
    prevents same-day returns from benefiting from weights that were only
    estimated at that rebalance.
    """
    decisions = decision_weights.reindex(returns.index).ffill()
    applied_weights = decisions.shift(1).fillna(0.0)

    gross_returns = (applied_weights * returns).sum(axis=1)
    turnover = applied_weights.diff().abs().sum(axis=1)
    turnover.iloc[0] = applied_weights.iloc[0].abs().sum()
    costs = turnover.fillna(0.0) * (tc_bps / 10000.0)
    net_returns = gross_returns - costs

    result = pd.DataFrame(
        {
            "gross_return": gross_returns,
            "turnover": turnover.fillna(0.0),
            "transaction_cost": costs,
            "net_return": net_returns,
        },
        index=returns.index,
    )
    active_period = applied_weights.abs().sum(axis=1).gt(1e-12).cummax()
    result.loc[~active_period, :] = pd.NA
    return result, applied_weights


def run_all_backtests(
    returns: pd.DataFrame,
    decisions: dict[str, pd.DataFrame],
    tc_bps: float = 5.0,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], pd.DataFrame]:
    backtests: dict[str, pd.DataFrame] = {}
    applied_weights: dict[str, pd.DataFrame] = {}
    daily_returns = pd.DataFrame(index=returns.index)

    for strategy, decision_weights in decisions.items():
        backtest, weights = run_backtest(returns, decision_weights, tc_bps=tc_bps)
        backtests[strategy] = backtest
        applied_weights[strategy] = weights
        daily_returns[strategy] = backtest["net_return"]

    return backtests, applied_weights, daily_returns
