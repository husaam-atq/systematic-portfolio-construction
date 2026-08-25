from __future__ import annotations

import numpy as np
import pandas as pd

from src.allocation import estimate_method
from src.backtest import (
    add_volatility_target_simulations,
    generate_target_weights,
    simulate_strategies,
)
from src.config import ResearchConfig
from src.metrics import performance_summary
from src.simulation import common_start_targets, delay_targets


SENSITIVITY_METHODS = [
    "Equal Weight",
    "Minimum Variance",
    "Maximum Sharpe",
    "Risk Parity",
    "Hierarchical Risk Parity",
]


def _summary_rows(
    dimension: str,
    value: str | float | int,
    simulations: dict,
    cash_returns: pd.Series,
) -> list[dict[str, object]]:
    summary = performance_summary(simulations, cash_returns=cash_returns).reset_index()
    rows = []
    for _, row in summary.iterrows():
        rows.append(
            {
                "dimension": dimension,
                "value": value,
                "strategy": row["strategy"],
                "cagr": row["cagr"],
                "sharpe_ratio": row["sharpe_ratio"],
                "annualized_volatility": row["annualized_volatility"],
                "max_drawdown": row["max_drawdown"],
                "average_monthly_turnover": row["average_monthly_turnover"],
                "effective_assets_by_weight": row["effective_assets_by_weight"],
            }
        )
    return rows


def parameter_sensitivity(
    asset_returns: pd.DataFrame,
    prices: pd.DataFrame,
    config: ResearchConfig,
    base_targets: dict[str, pd.DataFrame],
    base_simulations: dict,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    window_runs: list[tuple[int, dict]] = []
    for window in [126, 252, 504]:
        if window == config.estimation_window:
            targets = {strategy: base_targets[strategy] for strategy in SENSITIVITY_METHODS}
        else:
            targets, _, _ = generate_target_weights(
                asset_returns,
                prices,
                config,
                methods=SENSITIVITY_METHODS,
                estimation_window=window,
            )
        window_runs.append((window, simulate_strategies(asset_returns, targets, config)))

    common_window_start = max(
        simulation.performance["net_return"].first_valid_index()
        for _, simulations in window_runs
        for simulation in simulations.values()
    )
    for window, simulations in window_runs:
        restarted_targets = common_start_targets(
            simulations,
            asset_returns.columns,
            common_window_start,
        )
        aligned = simulate_strategies(
            asset_returns.loc[common_window_start:],
            restarted_targets,
            config,
        )
        rows.extend(
            _summary_rows(
                "estimation_window",
                window,
                aligned,
                asset_returns.loc[common_window_start:, config.cash_proxy],
            )
        )

    for frequency in ["monthly", "quarterly"]:
        if frequency == config.rebalance_frequency:
            targets = {strategy: base_targets[strategy] for strategy in SENSITIVITY_METHODS}
        else:
            targets, _, _ = generate_target_weights(
                asset_returns,
                prices,
                config,
                methods=SENSITIVITY_METHODS,
                rebalance_frequency=frequency,
            )
        rows.extend(_summary_rows("rebalance_frequency", frequency, simulate_strategies(asset_returns, targets, config), asset_returns[config.cash_proxy]))

    shrinkage_methods = [
        "Minimum Variance Shrinkage",
        "Maximum Sharpe Shrinkage",
        "Risk Parity Shrinkage",
    ]
    for shrinkage in [0.0, 0.25, 0.50, 0.75]:
        if shrinkage == config.covariance_shrinkage:
            targets = {strategy: base_targets[strategy] for strategy in shrinkage_methods}
        else:
            targets, _, _ = generate_target_weights(
                asset_returns,
                prices,
                config,
                methods=shrinkage_methods,
                shrinkage=shrinkage,
            )
        rows.extend(_summary_rows("covariance_shrinkage", shrinkage, simulate_strategies(asset_returns, targets, config), asset_returns[config.cash_proxy]))

    for cap in [0.30, 0.40, 0.60]:
        cap_methods = ["Minimum Variance", "Maximum Sharpe", "Risk Parity", "Hierarchical Risk Parity"]
        if cap == config.max_asset_weight:
            targets = {strategy: base_targets[strategy] for strategy in cap_methods}
        else:
            targets, _, _ = generate_target_weights(
                asset_returns,
                prices,
                config,
                methods=cap_methods,
                max_asset_weight=cap,
            )
        rows.extend(_summary_rows("maximum_asset_weight", cap, simulate_strategies(asset_returns, targets, config), asset_returns[config.cash_proxy]))

    for cost in [0.0, 5.0, 10.0, 20.0]:
        selected_targets = {strategy: base_targets[strategy] for strategy in SENSITIVITY_METHODS}
        simulations = simulate_strategies(asset_returns, selected_targets, config, transaction_cost_bps=cost)
        rows.extend(_summary_rows("transaction_cost_bps", cost, simulations, asset_returns[config.cash_proxy]))

    for target_volatility in [0.08, 0.10, 0.12]:
        simulations, _ = add_volatility_target_simulations(
            dict(base_simulations),
            asset_returns,
            config,
            target_volatility=target_volatility,
        )
        selected = {
            strategy: simulation
            for strategy, simulation in simulations.items()
            if f"Vol Target {target_volatility:.0%}" in strategy
        }
        rows.extend(_summary_rows("volatility_target", target_volatility, selected, asset_returns[config.cash_proxy]))

    for trend_days in [150, 200, 250]:
        targets, _, _ = generate_target_weights(
            asset_returns,
            prices,
            config,
            methods=["Trend Filtered Equal Weight", "Trend Filtered Minimum Variance", "Trend Filtered Risk Parity"],
            trend_moving_average_days=trend_days,
        )
        rows.extend(_summary_rows("trend_moving_average_days", trend_days, simulate_strategies(asset_returns, targets, config), asset_returns[config.cash_proxy]))

    for momentum_days in [189, 252, 315]:
        targets, _, _ = generate_target_weights(
            asset_returns,
            prices,
            config,
            methods=["Dual Momentum Equal Weight", "Dual Momentum Inverse Volatility"],
            momentum_lookback_days=momentum_days,
        )
        rows.extend(_summary_rows("momentum_lookback_days", momentum_days, simulate_strategies(asset_returns, targets, config), asset_returns[config.cash_proxy]))

    tactical = [
        "Trend Filtered Equal Weight",
        "Trend Filtered Minimum Variance",
        "Trend Filtered Risk Parity",
        "Dual Momentum Equal Weight",
        "Dual Momentum Inverse Volatility",
    ]
    for delay in [0, 1]:
        delayed_targets = {
            strategy: delay_targets(base_targets[strategy], asset_returns.index, delay)
            for strategy in tactical
        }
        rows.extend(_summary_rows("extra_execution_delay_days", delay, simulate_strategies(asset_returns, delayed_targets, config), asset_returns[config.cash_proxy]))

    for estimator in ["sample", "ewma", "grand_mean_shrink"]:
        targets, _, _ = generate_target_weights(
            asset_returns,
            prices,
            config,
            methods=["Maximum Sharpe", "Maximum Sharpe Shrinkage"],
            expected_return_estimator=estimator,
        )
        rows.extend(_summary_rows("expected_return_estimator", estimator, simulate_strategies(asset_returns, targets, config), asset_returns[config.cash_proxy]))

    return pd.DataFrame(rows)


def input_perturbation_sensitivity(
    asset_returns: pd.DataFrame,
    decision_dates: pd.DatetimeIndex,
    config: ResearchConfig,
    perturbations: int = 10,
    sampled_dates: int = 12,
) -> pd.DataFrame:
    methods = [
        "Minimum Variance",
        "Minimum Variance Shrinkage",
        "Maximum Sharpe",
        "Maximum Sharpe Shrinkage",
        "Risk Parity",
        "Risk Parity Shrinkage",
        "Hierarchical Risk Parity",
    ]
    valid_dates = [
        date
        for date in decision_dates
        if asset_returns.index.get_loc(date) + 1 >= config.estimation_window
    ]
    if len(valid_dates) > sampled_dates:
        positions = np.linspace(0, len(valid_dates) - 1, sampled_dates, dtype=int)
        valid_dates = [valid_dates[position] for position in positions]

    random = np.random.default_rng(config.bootstrap_seed)
    rows = []
    for date in valid_dates:
        position = asset_returns.index.get_loc(date)
        trailing = asset_returns.iloc[position - config.estimation_window + 1 : position + 1]
        daily_scale = trailing.std(ddof=0)
        for method in methods:
            base = estimate_method(
                method,
                trailing,
                max_weight=config.max_asset_weight,
                shrinkage=config.covariance_shrinkage,
                risk_free_asset=config.cash_proxy,
            ).weights
            distances = []
            for _ in range(perturbations):
                noise = random.normal(size=trailing.shape) * daily_scale.to_numpy() * 0.01
                perturbed = trailing + noise
                weights = estimate_method(
                    method,
                    perturbed,
                    max_weight=config.max_asset_weight,
                    shrinkage=config.covariance_shrinkage,
                    risk_free_asset=config.cash_proxy,
                ).weights
                distances.append(float((weights - base).abs().sum()))
            rows.append(
                {
                    "date": date,
                    "strategy": method,
                    "perturbation_scale_fraction_of_daily_volatility": 0.01,
                    "mean_l1_weight_change": float(np.mean(distances)),
                    "maximum_l1_weight_change": float(np.max(distances)),
                    "perturbations": perturbations,
                }
            )
    return pd.DataFrame(rows)


def optimizer_stability_summary(
    diagnostics: pd.DataFrame,
    simulations: dict,
    perturbations: pd.DataFrame,
) -> pd.DataFrame:
    diagnostic_summary = (
        diagnostics.groupby("strategy", as_index=False)
        .agg(
            solver_success_rate=("solver_success", "mean"),
            median_covariance_condition_number=("condition_number", "median"),
            maximum_covariance_condition_number=("condition_number", "max"),
            average_bound_count=("bound_count", "mean"),
            average_weight_dispersion=("weight_dispersion", "mean"),
            average_effective_assets=("effective_assets", "mean"),
            average_max_weight=("max_weight", "mean"),
        )
    )
    perturbation_summary = (
        perturbations.groupby("strategy", as_index=False)
        .agg(
            average_l1_weight_change=("mean_l1_weight_change", "mean"),
            worst_l1_weight_change=("maximum_l1_weight_change", "max"),
        )
    )
    performance = performance_summary(simulations).reset_index()[
        ["strategy", "average_monthly_turnover", "effective_assets_by_weight"]
    ]
    return diagnostic_summary.merge(perturbation_summary, on="strategy", how="left").merge(
        performance, on="strategy", how="left"
    )
