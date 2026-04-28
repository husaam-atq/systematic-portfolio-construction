from __future__ import annotations

import numpy as np
import pandas as pd

from src.allocation import (
    STRATEGY_NAMES,
    equal_weight,
    inverse_volatility,
    maximum_sharpe,
    minimum_variance,
    risk_parity,
    estimate_allocation_weights,
)
from src.risk import component_risk_contribution


def month_end_trading_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Return the last available trading date in each calendar month."""
    dates = pd.Series(index=index, data=index)
    return pd.DatetimeIndex(dates.groupby([index.year, index.month]).tail(1).to_list())


def generate_weight_decisions(
    returns: pd.DataFrame,
    prices: pd.DataFrame | None = None,
    estimation_window: int = 252,
    max_weight: float = 0.40,
    shrinkage: float = 0.25,
    cash_proxy: str = "SHY",
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """
    Estimate monthly allocation weights using only trailing returns.

    For a rebalance date at position i, the estimation window is
    returns.iloc[i - estimation_window:i], which excludes the rebalance date
    return and all future returns. The resulting decision is applied by
    run_backtest starting on the next trading session.
    """
    rebalance_dates = month_end_trading_dates(returns.index)
    trend_strategies = [
        "Trend Filtered Equal Weight",
        "Trend Filtered Minimum Variance",
        "Trend Filtered Risk Parity",
    ]
    dual_momentum_strategies = [
        "Dual Momentum Equal Weight",
        "Dual Momentum Inverse Vol",
    ]
    all_strategies = [*STRATEGY_NAMES, *trend_strategies, *dual_momentum_strategies]
    decisions = {
        strategy: pd.DataFrame(index=rebalance_dates, columns=returns.columns, dtype=float)
        for strategy in all_strategies
    }
    risk_records: list[dict[str, float | str | pd.Timestamp]] = []
    trend_signal = trend_eligibility(prices) if prices is not None else None
    momentum_signal = dual_momentum_signal(prices) if prices is not None else None

    for rebalance_date in rebalance_dates:
        position = returns.index.get_loc(rebalance_date)
        if position < estimation_window:
            continue

        trailing_returns = returns.iloc[position - estimation_window:position].dropna(how="any")
        if len(trailing_returns) < estimation_window:
            continue

        estimated = estimate_allocation_weights(
            trailing_returns,
            max_weight=max_weight,
            shrinkage=shrinkage,
        )

        if trend_signal is not None:
            trend_weights = estimate_trend_filtered_weights(
                trailing_returns=trailing_returns,
                trend_row=trend_signal.reindex(returns.index).loc[rebalance_date],
                max_weight=max_weight,
                cash_proxy=cash_proxy,
            )
            estimated.update(trend_weights)

        if momentum_signal is not None:
            momentum_weights = estimate_dual_momentum_weights(
                trailing_returns=trailing_returns,
                momentum_row=momentum_signal.reindex(returns.index).loc[rebalance_date],
                max_weight=max_weight,
                cash_proxy=cash_proxy,
            )
            estimated.update(momentum_weights)

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


def trend_eligibility(
    prices: pd.DataFrame,
    lookback: int = 252,
    defensive_assets: tuple[str, ...] = ("SHY", "IEF", "TLT"),
) -> pd.DataFrame:
    """
    Build lagged trend eligibility from prices.

    The signal uses the prior trading day's close relative to its 200-day
    moving average, so a rebalance never uses the same close that would be
    needed to trade the rebalance date return.
    """
    lagged_prices = prices.shift(1)
    moving_average = lagged_prices.rolling(200).mean()
    trend = lagged_prices > moving_average
    trend = trend.where(lagged_prices.rolling(lookback).count() >= lookback)
    for asset in defensive_assets:
        if asset in trend.columns:
            trend[asset] = True
    return trend.fillna(False)


def dual_momentum_signal(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Trailing 12-month momentum excluding the most recent month, lagged one day.

    At date t this uses prices through t-1 and excludes approximately the most
    recent 21 trading days, preventing same-day close information from entering
    a rebalance.
    """
    lagged_prices = prices.shift(1)
    return lagged_prices.shift(21) / lagged_prices.shift(252) - 1.0


def _zero_weight(index: pd.Index) -> pd.Series:
    return pd.Series(0.0, index=index)


def _expand_weights(weights: pd.Series, columns: pd.Index) -> pd.Series:
    expanded = _zero_weight(columns)
    expanded.loc[weights.index] = weights
    return expanded


def estimate_trend_filtered_weights(
    trailing_returns: pd.DataFrame,
    trend_row: pd.Series,
    max_weight: float = 0.40,
    cash_proxy: str = "SHY",
) -> dict[str, pd.Series]:
    eligible_assets = trend_row[trend_row.fillna(False)].index.intersection(trailing_returns.columns)
    if len(eligible_assets) == 0 and cash_proxy in trailing_returns.columns:
        eligible_assets = pd.Index([cash_proxy])

    filtered_returns = trailing_returns.loc[:, eligible_assets].dropna(how="any")
    if filtered_returns.empty:
        fallback = _zero_weight(trailing_returns.columns)
        if cash_proxy in fallback.index:
            fallback.loc[cash_proxy] = 1.0
        return {
            "Trend Filtered Equal Weight": fallback,
            "Trend Filtered Minimum Variance": fallback,
            "Trend Filtered Risk Parity": fallback,
        }

    feasible_max_weight = max(max_weight, 1.0 / filtered_returns.shape[1])
    return {
        "Trend Filtered Equal Weight": _expand_weights(
            equal_weight(filtered_returns, max_weight=feasible_max_weight),
            trailing_returns.columns,
        ),
        "Trend Filtered Minimum Variance": _expand_weights(
            minimum_variance(filtered_returns, max_weight=feasible_max_weight),
            trailing_returns.columns,
        ),
        "Trend Filtered Risk Parity": _expand_weights(
            risk_parity(filtered_returns, max_weight=feasible_max_weight),
            trailing_returns.columns,
        ),
    }


def estimate_dual_momentum_weights(
    trailing_returns: pd.DataFrame,
    momentum_row: pd.Series,
    max_weight: float = 0.40,
    cash_proxy: str = "SHY",
    top_n: int = 3,
) -> dict[str, pd.Series]:
    valid_momentum = momentum_row.dropna().reindex(trailing_returns.columns).dropna()
    selected = valid_momentum[valid_momentum > 0.0].sort_values(ascending=False).head(top_n).index
    weights_equal = _zero_weight(trailing_returns.columns)
    weights_inverse_vol = _zero_weight(trailing_returns.columns)

    if len(selected) > 0:
        selected_weight = min(len(selected), top_n) / top_n
        equal_selected = pd.Series(selected_weight / len(selected), index=selected)
        weights_equal.loc[selected] = equal_selected

        selected_returns = trailing_returns.loc[:, selected].dropna(how="any")
        feasible_max_weight = max(max_weight, 1.0 / selected_returns.shape[1])
        inv_selected = inverse_volatility(selected_returns, max_weight=feasible_max_weight)
        weights_inverse_vol.loc[selected] = inv_selected * selected_weight

    cash_weight = 1.0 - weights_equal.sum()
    if cash_proxy in weights_equal.index and cash_weight > 0.0:
        weights_equal.loc[cash_proxy] += cash_weight
        weights_inverse_vol.loc[cash_proxy] += 1.0 - weights_inverse_vol.sum()

    return {
        "Dual Momentum Equal Weight": weights_equal,
        "Dual Momentum Inverse Vol": weights_inverse_vol,
    }


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


def run_backtest_from_applied_weights(
    returns: pd.DataFrame,
    applied_weights: pd.DataFrame,
    tc_bps: float = 5.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    applied_weights = applied_weights.reindex(returns.index).fillna(0.0)
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


def apply_volatility_target(
    returns: pd.DataFrame,
    base_applied_weights: pd.DataFrame,
    target_volatility: float = 0.10,
    window: int = 63,
    max_leverage: float = 1.5,
    tc_bps: float = 5.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Scale an already lagged strategy using only past realised strategy returns.

    Raw realised volatility uses rolling base gross returns through t-1 because
    the scaling series is shifted by one day before being applied.
    """
    aligned_weights = base_applied_weights.reindex(returns.index).fillna(0.0)
    base_gross_returns = (aligned_weights * returns).sum(axis=1)
    active_period = aligned_weights.abs().sum(axis=1).gt(1e-12).cummax()
    base_gross_returns.loc[~active_period] = pd.NA
    realised_volatility = base_gross_returns.rolling(window).std(ddof=0) * np.sqrt(252)
    raw_scale = target_volatility / realised_volatility
    scale = raw_scale.clip(lower=0.0, upper=max_leverage).shift(1).fillna(0.0)
    scaled_weights = base_applied_weights.mul(scale, axis=0)
    result, weights = run_backtest_from_applied_weights(returns, scaled_weights, tc_bps=tc_bps)
    return result, weights, scale


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
