from __future__ import annotations

import numpy as np
import pandas as pd

from src.allocation import (
    AllocationResult,
    covariance_matrix,
    estimate_expected_returns,
    estimate_method,
    inverse_volatility,
)
from src.config import ResearchConfig, STATIC_METHODS, TACTICAL_METHODS, VOL_TARGET_BASE_METHODS
from src.risk import risk_snapshot
from src.signals import dual_momentum, trend_eligibility
from src.simulation import SimulationResult, simulate_portfolio


def rebalance_dates(index: pd.DatetimeIndex, frequency: str = "monthly") -> pd.DatetimeIndex:
    if frequency == "monthly":
        periods = index.to_period("M")
    elif frequency == "quarterly":
        periods = index.to_period("Q")
    else:
        raise ValueError("Rebalance frequency must be monthly or quarterly.")
    series = pd.Series(index, index=index)
    return pd.DatetimeIndex(series.groupby(periods).last().to_numpy())


def _diagnostic_row(
    date: pd.Timestamp,
    strategy: str,
    result: AllocationResult,
    input_start: pd.Timestamp,
    input_end: pd.Timestamp,
    observations: int,
) -> dict[str, object]:
    return {
        "date": date,
        "strategy": strategy,
        "input_start": input_start,
        "input_end": input_end,
        "estimation_observations": observations,
        **result.diagnostics,
    }


def _record_risk(
    date: pd.Timestamp,
    strategy: str,
    result: AllocationResult,
) -> list[dict[str, object]]:
    snapshot = risk_snapshot(result.weights, result.covariance)
    snapshot.insert(0, "strategy", strategy)
    snapshot.insert(0, "date", date)
    return snapshot.to_dict(orient="records")


def _trend_result(
    base_result: AllocationResult,
    eligibility: pd.Series,
    cash_proxy: str,
    method: str,
) -> AllocationResult:
    weights = base_result.weights.copy()
    risky_assets = [asset for asset in weights.index if asset != cash_proxy]
    ineligible = [asset for asset in risky_assets if not bool(eligibility.get(asset, False))]
    defensive_transfer = float(weights.loc[ineligible].sum()) if ineligible else 0.0
    weights.loc[ineligible] = 0.0
    weights.loc[cash_proxy] += defensive_transfer
    diagnostics = dict(base_result.diagnostics)
    diagnostics.update(
        {
            "method": method,
            "trend_ineligible_assets": len(ineligible),
            "defensive_transfer": defensive_transfer,
            "max_weight": float(weights.max()),
            "weight_hhi": float((weights**2).sum()),
            "effective_assets": float(1.0 / (weights**2).sum()),
            "weight_dispersion": float(weights.std(ddof=0)),
        }
    )
    return AllocationResult(weights, base_result.covariance, base_result.expected_returns, diagnostics)


def _dual_momentum_result(
    trailing_returns: pd.DataFrame,
    momentum_row: pd.Series,
    cash_proxy: str,
    top_n: int,
    inverse_vol: bool,
) -> AllocationResult:
    risky_assets = [asset for asset in trailing_returns.columns if asset != cash_proxy]
    ranked = momentum_row.reindex(risky_assets).dropna()
    selected = ranked[ranked > 0.0].sort_values(ascending=False).head(top_n).index.tolist()
    weights = pd.Series(0.0, index=trailing_returns.columns)
    risky_budget = len(selected) / top_n
    if selected:
        if inverse_vol:
            selected_weights = inverse_volatility(
                trailing_returns[selected],
                max_weight=max(1.0 / len(selected), 0.40),
            )
            weights.loc[selected] = selected_weights * risky_budget
        else:
            weights.loc[selected] = 1.0 / top_n
    weights.loc[cash_proxy] = 1.0 - weights.sum()
    covariance = covariance_matrix(trailing_returns)
    hhi = float((weights**2).sum())
    diagnostics: dict[str, float | int | bool | str] = {
        "method": "dual_momentum_inverse_volatility" if inverse_vol else "dual_momentum_equal_weight",
        "solver_success": True,
        "solver_status": "rule based",
        "condition_number": float(np.linalg.cond(covariance.to_numpy())),
        "bound_count": 0,
        "max_weight": float(weights.max()),
        "weight_hhi": hhi,
        "effective_assets": float(1.0 / hhi),
        "weight_dispersion": float(weights.std(ddof=0)),
        "selected_assets": len(selected),
    }
    return AllocationResult(weights, covariance, estimate_expected_returns(trailing_returns), diagnostics)


def generate_target_weights(
    asset_returns: pd.DataFrame,
    prices: pd.DataFrame,
    config: ResearchConfig,
    methods: list[str] | None = None,
    estimation_window: int | None = None,
    rebalance_frequency: str | None = None,
    max_asset_weight: float | None = None,
    shrinkage: float | None = None,
    expected_return_estimator: str = "sample",
    trend_moving_average_days: int | None = None,
    momentum_lookback_days: int | None = None,
    momentum_skip_days: int | None = None,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    """Create close-t targets using data through t; simulation applies them after return t."""
    selected_methods = methods or [*STATIC_METHODS, *TACTICAL_METHODS]
    window = estimation_window or config.estimation_window
    frequency = rebalance_frequency or config.rebalance_frequency
    cap = max_asset_weight or config.max_asset_weight
    shrink = config.covariance_shrinkage if shrinkage is None else shrinkage
    trend_days = trend_moving_average_days or config.trend_moving_average_days
    momentum_days = momentum_lookback_days or config.momentum_lookback_days
    skip_days = momentum_skip_days or config.momentum_skip_days
    dates = rebalance_dates(asset_returns.index, frequency)
    targets = {
        method: pd.DataFrame(index=dates, columns=asset_returns.columns, dtype=float)
        for method in selected_methods
    }
    diagnostics: list[dict[str, object]] = []
    risk_records: list[dict[str, object]] = []
    trend = trend_eligibility(prices, moving_average_days=trend_days, cash_proxy=config.cash_proxy)
    momentum = dual_momentum(prices, lookback_days=momentum_days, skip_days=skip_days)

    for date in dates:
        position = asset_returns.index.get_loc(date)
        if position + 1 < window:
            continue
        trailing = asset_returns.iloc[position - window + 1 : position + 1]
        if len(trailing) != window or trailing.index.max() != date:
            continue

        results: dict[str, AllocationResult] = {}
        for method in [method for method in selected_methods if method in STATIC_METHODS]:
            results[method] = estimate_method(
                method,
                trailing,
                max_weight=cap,
                shrinkage=shrink,
                expected_return_estimator=expected_return_estimator,
                risk_free_asset=config.cash_proxy,
            )

        if "Trend Filtered Equal Weight" in selected_methods:
            base = estimate_method("Equal Weight", trailing, max_weight=cap)
            results["Trend Filtered Equal Weight"] = _trend_result(
                base, trend.loc[date], config.cash_proxy, "trend_filtered_equal_weight"
            )
        if "Trend Filtered Minimum Variance" in selected_methods:
            base = estimate_method("Minimum Variance", trailing, max_weight=cap)
            results["Trend Filtered Minimum Variance"] = _trend_result(
                base, trend.loc[date], config.cash_proxy, "trend_filtered_minimum_variance"
            )
        if "Trend Filtered Risk Parity" in selected_methods:
            base = estimate_method("Risk Parity", trailing, max_weight=cap)
            results["Trend Filtered Risk Parity"] = _trend_result(
                base, trend.loc[date], config.cash_proxy, "trend_filtered_risk_parity"
            )
        if "Dual Momentum Equal Weight" in selected_methods:
            results["Dual Momentum Equal Weight"] = _dual_momentum_result(
                trailing,
                momentum.loc[date],
                config.cash_proxy,
                config.momentum_top_n,
                inverse_vol=False,
            )
        if "Dual Momentum Inverse Volatility" in selected_methods:
            results["Dual Momentum Inverse Volatility"] = _dual_momentum_result(
                trailing,
                momentum.loc[date],
                config.cash_proxy,
                config.momentum_top_n,
                inverse_vol=True,
            )

        for strategy, result in results.items():
            targets[strategy].loc[date] = result.weights
            diagnostics.append(
                _diagnostic_row(date, strategy, result, trailing.index.min(), trailing.index.max(), len(trailing))
            )
            risk_records.extend(_record_risk(date, strategy, result))

    targets = {strategy: frame.dropna(how="all") for strategy, frame in targets.items()}
    return targets, pd.DataFrame(diagnostics), pd.DataFrame(risk_records)


def simulate_strategies(
    asset_returns: pd.DataFrame,
    targets: dict[str, pd.DataFrame],
    config: ResearchConfig,
    transaction_cost_bps: float | None = None,
) -> dict[str, SimulationResult]:
    cash_returns = asset_returns[config.cash_proxy]
    cost = config.transaction_cost_bps if transaction_cost_bps is None else transaction_cost_bps
    return {
        strategy: simulate_portfolio(
            asset_returns,
            target,
            cash_returns,
            transaction_cost_bps=cost,
            financing_spread_annual=config.financing_spread_annual,
            max_leverage=config.volatility_max_leverage,
        )
        for strategy, target in targets.items()
    }


def volatility_target_weights(
    base_simulation: SimulationResult,
    asset_returns: pd.DataFrame,
    target_volatility: float,
    window: int,
    max_leverage: float,
) -> tuple[pd.DataFrame, pd.Series]:
    """Create daily close-t exposure targets from a one-day-lagged realised-vol estimate."""
    gross = base_simulation.performance["gross_return"]
    realised_volatility = gross.rolling(window, min_periods=window).std(ddof=0) * np.sqrt(252.0)
    scale = (target_volatility / realised_volatility).clip(0.0, max_leverage).shift(1)
    base_assets = base_simulation.post_trade_weights[asset_returns.columns]
    targets = base_assets.mul(scale, axis=0).dropna(how="all")
    return targets, scale


def add_volatility_target_simulations(
    simulations: dict[str, SimulationResult],
    asset_returns: pd.DataFrame,
    config: ResearchConfig,
    target_volatility: float | None = None,
) -> tuple[dict[str, SimulationResult], pd.DataFrame]:
    target = config.volatility_target if target_volatility is None else target_volatility
    metadata = []
    additions: dict[str, SimulationResult] = {}
    for base_method in VOL_TARGET_BASE_METHODS:
        if base_method not in simulations:
            continue
        target_weights, scale = volatility_target_weights(
            simulations[base_method],
            asset_returns,
            target,
            config.volatility_window,
            config.volatility_max_leverage,
        )
        strategy = f"{base_method} Vol Target {target:.0%}"
        additions[strategy] = simulate_portfolio(
            asset_returns,
            target_weights,
            asset_returns[config.cash_proxy],
            transaction_cost_bps=config.transaction_cost_bps,
            financing_spread_annual=config.financing_spread_annual,
            max_leverage=config.volatility_max_leverage,
        )
        metadata.append(
            {
                "strategy": strategy,
                "base_strategy": base_method,
                "target_volatility": target,
                "average_scale": float(scale.dropna().mean()),
                "maximum_scale": float(scale.dropna().max()),
                "fraction_above_one_x": float((scale.dropna() > 1.0).mean()),
            }
        )
    simulations.update(additions)
    return simulations, pd.DataFrame(metadata)


def daily_return_frame(simulations: dict[str, SimulationResult], field: str = "net_return") -> pd.DataFrame:
    return pd.DataFrame(
        {strategy: simulation.performance[field] for strategy, simulation in simulations.items()}
    )


def daily_weight_frame(simulations: dict[str, SimulationResult]) -> pd.DataFrame:
    frames = {strategy: result.post_trade_weights for strategy, result in simulations.items()}
    combined = pd.concat(frames, axis=1)
    combined.columns = [f"{strategy}__{asset}" for strategy, asset in combined.columns]
    return combined
