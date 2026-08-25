from __future__ import annotations

import numpy as np
import pandas as pd

from src.simulation import SimulationResult


TRADING_DAYS = 252

STRESS_EPISODES: dict[str, tuple[str, str]] = {
    "Euro-area / sovereign stress": ("2011-07-01", "2011-10-04"),
    "2013 rate shock": ("2013-05-02", "2013-09-05"),
    "2015-16 growth / commodity stress": ("2015-07-01", "2016-02-11"),
    "Q4 2018 equity sell-off": ("2018-10-01", "2018-12-24"),
    "COVID shock": ("2020-02-19", "2020-03-23"),
    "2022 inflation / rate shock": ("2022-01-03", "2022-10-14"),
    "2023-24 recovery / rate regime": ("2023-01-03", "2024-12-31"),
}

DEVELOPMENT_SEGMENTS: dict[str, tuple[str, str]] = {
    "2011-2014": ("2011-01-01", "2014-12-31"),
    "2015-2017": ("2015-01-01", "2017-12-31"),
    "2018-2020": ("2018-01-01", "2020-12-31"),
    "2021-2022": ("2021-01-01", "2022-12-31"),
    "2023-2024": ("2023-01-01", "2024-12-31"),
}


def total_return(returns: pd.Series) -> float:
    values = returns.dropna()
    return float((1.0 + values).prod() - 1.0) if not values.empty else np.nan


def cagr(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    values = returns.dropna()
    if values.empty:
        return np.nan
    growth = float((1.0 + values).prod())
    years = len(values) / periods_per_year
    return float(growth ** (1.0 / years) - 1.0) if growth > 0 and years > 0 else np.nan


def annualized_volatility(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    values = returns.dropna()
    return float(values.std(ddof=0) * np.sqrt(periods_per_year)) if not values.empty else np.nan


def sharpe_ratio(
    returns: pd.Series,
    periods_per_year: int = TRADING_DAYS,
    risk_free_returns: pd.Series | None = None,
) -> float:
    if risk_free_returns is None:
        values = returns.dropna()
    else:
        aligned = pd.concat([returns, risk_free_returns], axis=1, join="inner").dropna()
        values = aligned.iloc[:, 0] - aligned.iloc[:, 1]
    standard_deviation = values.std(ddof=0)
    return (
        float(values.mean() / standard_deviation * np.sqrt(periods_per_year))
        if len(values) and standard_deviation > 0
        else np.nan
    )


def downside_deviation(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    values = returns.dropna()
    downside = np.minimum(values.to_numpy(), 0.0)
    return float(np.sqrt(np.mean(downside**2)) * np.sqrt(periods_per_year)) if len(values) else np.nan


def sortino_ratio(
    returns: pd.Series,
    periods_per_year: int = TRADING_DAYS,
    risk_free_returns: pd.Series | None = None,
) -> float:
    if risk_free_returns is None:
        values = returns.dropna()
    else:
        aligned = pd.concat([returns, risk_free_returns], axis=1, join="inner").dropna()
        values = aligned.iloc[:, 0] - aligned.iloc[:, 1]
    downside = downside_deviation(values, periods_per_year)
    return float(values.mean() * periods_per_year / downside) if downside and downside > 0 else np.nan


def drawdown_series(returns: pd.Series) -> pd.Series:
    values = returns.dropna()
    equity = (1.0 + values).cumprod()
    return equity / equity.cummax() - 1.0


def max_drawdown(returns: pd.Series) -> float:
    drawdowns = drawdown_series(returns)
    return float(drawdowns.min()) if not drawdowns.empty else np.nan


def max_drawdown_duration(returns: pd.Series) -> int:
    drawdowns = drawdown_series(returns)
    current = maximum = 0
    for drawdown in drawdowns:
        current = current + 1 if drawdown < 0 else 0
        maximum = max(maximum, current)
    return maximum


def monthly_returns(returns: pd.Series) -> pd.Series:
    values = returns.dropna()
    return (1.0 + values).resample("ME").prod() - 1.0 if not values.empty else pd.Series(dtype=float)


def _weight_metrics(simulation: SimulationResult) -> tuple[float, float, float]:
    active = simulation.post_trade_weights.loc[simulation.performance["active"] > 0.0]
    assets = active.drop(columns="CASH", errors="ignore")
    if assets.empty:
        return np.nan, np.nan, np.nan
    gross = assets.abs().sum(axis=1).replace(0.0, np.nan)
    normalized = assets.div(gross, axis=0)
    hhi = (normalized**2).sum(axis=1)
    return float(hhi.mean()), float((1.0 / hhi.replace(0.0, np.nan)).mean()), float(normalized.max(axis=1).mean())


def performance_summary(
    simulations: dict[str, SimulationResult],
    cash_returns: pd.Series | None = None,
) -> pd.DataFrame:
    rows = []
    for strategy, simulation in simulations.items():
        performance = simulation.performance
        returns = performance["net_return"].dropna()
        monthly = monthly_returns(returns)
        maximum_drawdown = max_drawdown(returns)
        weight_hhi, effective_assets, average_max_weight = _weight_metrics(simulation)
        monthly_turnover = performance["turnover"].resample("ME").sum(min_count=1)
        monthly_cost = performance["transaction_cost_return"].resample("ME").sum(min_count=1)
        rows.append(
            {
                "strategy": strategy,
                "total_return": total_return(returns),
                "cagr": cagr(returns),
                "annualized_volatility": annualized_volatility(returns),
                "sharpe_ratio": sharpe_ratio(returns, risk_free_returns=cash_returns),
                "zero_hurdle_sharpe_ratio": sharpe_ratio(returns),
                "sortino_ratio": sortino_ratio(returns, risk_free_returns=cash_returns),
                "calmar_ratio": cagr(returns) / abs(maximum_drawdown) if maximum_drawdown < 0 else np.nan,
                "max_drawdown": maximum_drawdown,
                "average_drawdown": float(drawdown_series(returns).clip(upper=0.0).mean()),
                "max_drawdown_duration_days": max_drawdown_duration(returns),
                "downside_deviation": downside_deviation(returns),
                "monthly_win_rate": float((monthly > 0.0).mean()) if len(monthly) else np.nan,
                "daily_hit_rate": float((returns > 0.0).mean()) if len(returns) else np.nan,
                "average_monthly_turnover": float(monthly_turnover.mean()),
                "average_monthly_transaction_cost": float(monthly_cost.mean()),
                "total_transaction_cost": float(performance["transaction_cost_return"].sum()),
                "best_month": float(monthly.max()) if len(monthly) else np.nan,
                "worst_month": float(monthly.min()) if len(monthly) else np.nan,
                "average_weight_hhi": weight_hhi,
                "effective_assets_by_weight": effective_assets,
                "average_maximum_asset_weight": average_max_weight,
                "average_gross_exposure": float(performance["gross_exposure"].dropna().mean()),
                "maximum_gross_exposure": float(performance["gross_exposure"].dropna().max()),
                "average_cash_weight": float(performance["cash_weight"].dropna().mean()),
                "observations": int(len(returns)),
            }
        )
    return pd.DataFrame(rows).set_index("strategy")


def benchmark_relative_metrics(
    strategy_returns: pd.DataFrame,
    spy_returns: pd.Series,
    benchmarks: tuple[str, ...] = ("Equal Weight", "Traditional 60/40"),
) -> pd.DataFrame:
    rows = []
    references: dict[str, pd.Series] = {
        benchmark: strategy_returns[benchmark] for benchmark in benchmarks if benchmark in strategy_returns
    }
    references["SPY informational"] = spy_returns
    for strategy in strategy_returns.columns:
        for benchmark, reference in references.items():
            aligned = pd.concat([strategy_returns[strategy], reference], axis=1, join="inner").dropna()
            if aligned.empty:
                continue
            aligned.columns = ["strategy", "benchmark"]
            excess = aligned["strategy"] - aligned["benchmark"]
            tracking_error = excess.std(ddof=0) * np.sqrt(TRADING_DAYS)
            benchmark_variance = aligned["benchmark"].var(ddof=0)
            beta = aligned["strategy"].cov(aligned["benchmark"], ddof=0) / benchmark_variance if benchmark_variance > 0 else np.nan
            alpha = (aligned["strategy"].mean() - beta * aligned["benchmark"].mean()) * TRADING_DAYS
            rows.append(
                {
                    "strategy": strategy,
                    "benchmark": benchmark,
                    "annualized_alpha": float(alpha),
                    "annualized_excess_return": float(excess.mean() * TRADING_DAYS),
                    "tracking_error": float(tracking_error),
                    "information_ratio": float(excess.mean() / excess.std(ddof=0) * np.sqrt(TRADING_DAYS))
                    if excess.std(ddof=0) > 0
                    else np.nan,
                    "beta": float(beta),
                    "total_return_difference": total_return(aligned["strategy"]) - total_return(aligned["benchmark"]),
                }
            )
    return pd.DataFrame(rows)


def capture_ratios(strategy_returns: pd.DataFrame, spy_returns: pd.Series) -> pd.DataFrame:
    rows = []
    for strategy in strategy_returns.columns:
        aligned = pd.concat([strategy_returns[strategy], spy_returns], axis=1, join="inner").dropna()
        aligned.columns = ["strategy", "spy"]
        strategy_monthly = monthly_returns(aligned["strategy"])
        spy_monthly = monthly_returns(aligned["spy"])
        monthly = pd.concat([strategy_monthly, spy_monthly], axis=1).dropna()
        monthly.columns = ["strategy", "spy"]
        up = monthly["spy"] > 0.0
        down = monthly["spy"] < 0.0
        rows.append(
            {
                "strategy": strategy,
                "upside_capture_vs_spy": float(monthly.loc[up, "strategy"].mean() / monthly.loc[up, "spy"].mean())
                if up.any()
                else np.nan,
                "downside_capture_vs_spy": float(monthly.loc[down, "strategy"].mean() / monthly.loc[down, "spy"].mean())
                if down.any()
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def stress_results(strategy_returns: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for episode, (start, end) in STRESS_EPISODES.items():
        for strategy in strategy_returns.columns:
            values = strategy_returns.loc[start:end, strategy].dropna()
            rows.append(
                {
                    "episode": episode,
                    "start": start,
                    "end": end,
                    "strategy": strategy,
                    "total_return": total_return(values),
                    "annualized_volatility": annualized_volatility(values),
                    "max_drawdown": max_drawdown(values),
                    "observations": len(values),
                }
            )
    return pd.DataFrame(rows)


def segment_results(
    simulations: dict[str, SimulationResult],
    risk_records: pd.DataFrame,
    segments: dict[str, tuple[str, str]] = DEVELOPMENT_SEGMENTS,
    cash_returns: pd.Series | None = None,
) -> pd.DataFrame:
    rows = []
    equal_returns = simulations["Equal Weight"].performance["net_return"]
    for segment, (start, end) in segments.items():
        for strategy, simulation in simulations.items():
            performance = simulation.performance.loc[start:end]
            returns = performance["net_return"].dropna()
            weights = simulation.post_trade_weights.loc[start:end].drop(columns="CASH", errors="ignore")
            active_weights = weights.loc[performance["active"].fillna(0.0) > 0.0]
            risk_slice = risk_records[
                (risk_records["strategy"] == strategy)
                & (risk_records["date"] >= start)
                & (risk_records["date"] <= end)
            ]
            risk_concentration = (
                risk_slice.groupby("date")["risk_concentration_hhi"].first().mean()
                if not risk_slice.empty
                else np.nan
            )
            aligned = pd.concat([returns, equal_returns.loc[start:end]], axis=1).dropna()
            rows.append(
                {
                    "segment": segment,
                    "strategy": strategy,
                    "total_return": total_return(returns),
                    "cagr": cagr(returns),
                    "sharpe_ratio": sharpe_ratio(
                        returns,
                        risk_free_returns=cash_returns.loc[start:end] if cash_returns is not None else None,
                    ),
                    "annualized_volatility": annualized_volatility(returns),
                    "max_drawdown": max_drawdown(returns),
                    "average_monthly_turnover": float(performance["turnover"].resample("ME").sum(min_count=1).mean()),
                    "average_weight_hhi": float((active_weights**2).sum(axis=1).mean()) if not active_weights.empty else np.nan,
                    "average_risk_concentration_hhi": float(risk_concentration),
                    "return_difference_vs_equal_weight": total_return(aligned.iloc[:, 0]) - total_return(aligned.iloc[:, 1])
                    if not aligned.empty
                    else np.nan,
                    "observations": len(returns),
                }
            )
    return pd.DataFrame(rows)


def moving_block_bootstrap(
    strategy_returns: pd.DataFrame,
    benchmark: str = "Equal Weight",
    block_days: int = 21,
    resamples: int = 500,
    seed: int = 20260825,
    cash_returns: pd.Series | None = None,
) -> pd.DataFrame:
    rows = []
    random = np.random.default_rng(seed)
    for strategy in strategy_returns.columns:
        pieces = [strategy_returns[strategy], strategy_returns[benchmark]]
        if cash_returns is not None:
            pieces.append(cash_returns.rename("cash_proxy"))
        aligned = pd.concat(pieces, axis=1, join="inner").dropna()
        if aligned.empty:
            continue
        values = aligned.to_numpy()
        n_observations = len(values)
        starts_upper = max(n_observations - block_days + 1, 1)
        bootstrap_metrics: list[list[float]] = []
        for _ in range(resamples):
            sampled_indices: list[int] = []
            while len(sampled_indices) < n_observations:
                start = int(random.integers(0, starts_upper))
                sampled_indices.extend(range(start, min(start + block_days, n_observations)))
            sampled = values[np.asarray(sampled_indices[:n_observations])]
            strategy_sample = pd.Series(sampled[:, 0])
            benchmark_sample = pd.Series(sampled[:, 1])
            cash_sample = pd.Series(sampled[:, 2]) if sampled.shape[1] > 2 else None
            bootstrap_metrics.append(
                [
                    cagr(strategy_sample),
                    sharpe_ratio(strategy_sample, risk_free_returns=cash_sample),
                    max_drawdown(strategy_sample),
                    float((strategy_sample - benchmark_sample).mean() * TRADING_DAYS),
                    sharpe_ratio(strategy_sample, risk_free_returns=cash_sample)
                    - sharpe_ratio(benchmark_sample, risk_free_returns=cash_sample),
                ]
            )
        samples = np.asarray(bootstrap_metrics)
        estimates = np.asarray(
            [
                cagr(aligned.iloc[:, 0]),
                sharpe_ratio(
                    aligned.iloc[:, 0],
                    risk_free_returns=aligned.iloc[:, 2] if aligned.shape[1] > 2 else None,
                ),
                max_drawdown(aligned.iloc[:, 0]),
                float((aligned.iloc[:, 0] - aligned.iloc[:, 1]).mean() * TRADING_DAYS),
                sharpe_ratio(
                    aligned.iloc[:, 0],
                    risk_free_returns=aligned.iloc[:, 2] if aligned.shape[1] > 2 else None,
                )
                - sharpe_ratio(
                    aligned.iloc[:, 1],
                    risk_free_returns=aligned.iloc[:, 2] if aligned.shape[1] > 2 else None,
                ),
            ]
        )
        metric_names = ["cagr", "sharpe_ratio", "max_drawdown", "excess_return_vs_equal_weight", "sharpe_difference_vs_equal_weight"]
        for index, metric in enumerate(metric_names):
            rows.append(
                {
                    "strategy": strategy,
                    "metric": metric,
                    "estimate": estimates[index],
                    "lower_95": float(np.nanquantile(samples[:, index], 0.025)),
                    "upper_95": float(np.nanquantile(samples[:, index], 0.975)),
                    "probability_above_zero": float(np.nanmean(samples[:, index] > 0.0)),
                    "block_days": block_days,
                    "resamples": resamples,
                }
            )
    return pd.DataFrame(rows)
