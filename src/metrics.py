from __future__ import annotations

import numpy as np
import pandas as pd


TRADING_DAYS = 252


def total_return(returns: pd.Series) -> float:
    returns = returns.dropna()
    if returns.empty:
        return float("nan")
    return float((1.0 + returns).prod() - 1.0)


def cagr(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    returns = returns.dropna()
    if returns.empty:
        return float("nan")
    growth = (1.0 + returns).prod()
    years = len(returns) / periods_per_year
    if years <= 0:
        return float("nan")
    return float(growth ** (1.0 / years) - 1.0)


def annualized_volatility(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    returns = returns.dropna()
    if returns.empty:
        return float("nan")
    return float(returns.std(ddof=0) * np.sqrt(periods_per_year))


def downside_deviation(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    returns = returns.dropna()
    downside = returns[returns < 0.0]
    if downside.empty:
        return 0.0
    return float(np.sqrt((downside**2).mean()) * np.sqrt(periods_per_year))


def sharpe_ratio(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    returns = returns.dropna()
    volatility = returns.std(ddof=0)
    if returns.empty or volatility <= 0:
        return float("nan")
    return float((returns.mean() / volatility) * np.sqrt(periods_per_year))


def sortino_ratio(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    downside = downside_deviation(returns, periods_per_year=periods_per_year)
    if downside <= 0:
        return float("nan")
    return float((returns.dropna().mean() * periods_per_year) / downside)


def drawdown_series(returns: pd.Series) -> pd.Series:
    returns = returns.fillna(0.0)
    equity = (1.0 + returns).cumprod()
    return equity / equity.cummax() - 1.0


def max_drawdown(returns: pd.Series) -> float:
    returns = returns.dropna()
    if returns.empty:
        return float("nan")
    return float(drawdown_series(returns).min())


def monthly_returns(returns: pd.Series) -> pd.Series:
    returns = returns.dropna()
    if returns.empty:
        return pd.Series(dtype=float)
    return (1.0 + returns).resample("ME").prod() - 1.0


def concentration_hhi(weights: pd.DataFrame) -> pd.Series:
    active = weights[weights.sum(axis=1) > 0.0]
    if active.empty:
        return pd.Series(dtype=float)
    return (active**2).sum(axis=1)


def performance_summary(
    backtests: dict[str, pd.DataFrame],
    weights: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows = []
    for strategy, result in backtests.items():
        returns = result["net_return"].dropna()
        monthly = monthly_returns(returns)
        max_dd = max_drawdown(returns)
        hhi = concentration_hhi(weights[strategy])
        monthly_turnover = result["turnover"].resample("ME").sum()
        monthly_cost = result["transaction_cost"].resample("ME").sum()

        rows.append(
            {
                "strategy": strategy,
                "total_return": total_return(returns),
                "cagr": cagr(returns),
                "annualized_volatility": annualized_volatility(returns),
                "sharpe_ratio": sharpe_ratio(returns),
                "sortino_ratio": sortino_ratio(returns),
                "calmar_ratio": cagr(returns) / abs(max_dd) if max_dd < 0 else float("nan"),
                "max_drawdown": max_dd,
                "downside_deviation": downside_deviation(returns),
                "monthly_win_rate": float((monthly > 0.0).mean()) if not monthly.empty else float("nan"),
                "daily_hit_rate": float((returns > 0.0).mean()) if not returns.empty else float("nan"),
                "average_turnover": float(monthly_turnover.mean()) if not monthly_turnover.empty else 0.0,
                "average_transaction_cost": float(monthly_cost.mean()) if not monthly_cost.empty else 0.0,
                "best_month": float(monthly.max()) if not monthly.empty else float("nan"),
                "worst_month": float(monthly.min()) if not monthly.empty else float("nan"),
                "average_concentration_hhi": float(hhi.mean()) if not hhi.empty else float("nan"),
                "effective_number_of_assets": float(1.0 / hhi.mean()) if not hhi.empty and hhi.mean() > 0 else float("nan"),
                "observations": int(returns.shape[0]),
            }
        )

    return pd.DataFrame(rows).set_index("strategy")


def subperiod_summary(
    daily_returns: pd.DataFrame,
    subperiods: dict[str, tuple[str, str]],
) -> pd.DataFrame:
    rows = []
    for label, (start, end) in subperiods.items():
        period_returns = daily_returns.loc[start:end]
        for strategy in daily_returns.columns:
            returns = period_returns[strategy].dropna()
            rows.append(
                {
                    "subperiod": label,
                    "strategy": strategy,
                    "total_return": total_return(returns),
                    "cagr": cagr(returns),
                    "sharpe_ratio": sharpe_ratio(returns),
                    "max_drawdown": max_drawdown(returns),
                    "annualized_volatility": annualized_volatility(returns),
                    "observations": int(returns.shape[0]),
                }
            )
    return pd.DataFrame(rows)


def benchmark_relative_metrics(
    daily_returns: pd.DataFrame,
    asset_returns: pd.DataFrame,
    equal_weight_name: str = "Equal Weight",
    spy_ticker: str = "SPY",
    periods_per_year: int = TRADING_DAYS,
) -> pd.DataFrame:
    rows = []
    equal_weight_returns = daily_returns[equal_weight_name].dropna()
    spy_returns = asset_returns[spy_ticker].dropna() if spy_ticker in asset_returns.columns else pd.Series(dtype=float)

    for strategy in daily_returns.columns:
        strategy_returns = daily_returns[strategy].dropna()

        aligned_strategy, aligned_equal = strategy_returns.align(equal_weight_returns, join="inner")
        excess_equal = aligned_strategy - aligned_equal
        tracking_error = excess_equal.std(ddof=0) * np.sqrt(periods_per_year)
        information_ratio = (
            excess_equal.mean() / excess_equal.std(ddof=0) * np.sqrt(periods_per_year)
            if excess_equal.std(ddof=0) > 0
            else float("nan")
        )

        aligned_strategy_spy, aligned_spy = strategy_returns.align(spy_returns, join="inner")
        spy_variance = aligned_spy.var(ddof=0)
        beta = (
            aligned_strategy_spy.cov(aligned_spy, ddof=0) / spy_variance
            if spy_variance > 0
            else float("nan")
        )

        monthly_strategy = monthly_returns(aligned_strategy_spy)
        monthly_spy = monthly_returns(aligned_spy)
        monthly_strategy, monthly_spy = monthly_strategy.align(monthly_spy, join="inner")
        up_months = monthly_spy > 0.0
        down_months = monthly_spy < 0.0
        upside_capture = (
            monthly_strategy[up_months].mean() / monthly_spy[up_months].mean()
            if up_months.any() and monthly_spy[up_months].mean() != 0
            else float("nan")
        )
        downside_capture = (
            monthly_strategy[down_months].mean() / monthly_spy[down_months].mean()
            if down_months.any() and monthly_spy[down_months].mean() != 0
            else float("nan")
        )

        rows.append(
            {
                "strategy": strategy,
                "annualized_alpha_vs_equal_weight": float(excess_equal.mean() * periods_per_year),
                "tracking_error_vs_equal_weight": float(tracking_error),
                "information_ratio_vs_equal_weight": float(information_ratio),
                "beta_vs_spy": float(beta),
                "upside_capture_vs_spy": float(upside_capture),
                "downside_capture_vs_spy": float(downside_capture),
            }
        )

    return pd.DataFrame(rows)


def drawdown_duration(returns: pd.Series) -> int:
    drawdowns = drawdown_series(returns.dropna())
    max_duration = 0
    current_duration = 0
    for value in drawdowns:
        if value < 0.0:
            current_duration += 1
            max_duration = max(max_duration, current_duration)
        else:
            current_duration = 0
    return max_duration


def period_return(returns: pd.Series, start: str, end: str) -> float:
    period = returns.loc[start:end].dropna()
    return total_return(period)


def stress_results(daily_returns: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for strategy in daily_returns.columns:
        returns = daily_returns[strategy].dropna()
        monthly = monthly_returns(returns)
        rolling_3m = (1.0 + monthly).rolling(3).apply(np.prod, raw=True) - 1.0
        drawdowns = drawdown_series(returns)
        negative_drawdowns = drawdowns[drawdowns < 0.0]

        rows.append(
            {
                "strategy": strategy,
                "max_drawdown": max_drawdown(returns),
                "average_drawdown": float(negative_drawdowns.mean()) if not negative_drawdowns.empty else 0.0,
                "max_drawdown_duration_days": drawdown_duration(returns),
                "worst_1_month_return": float(monthly.min()) if not monthly.empty else float("nan"),
                "worst_3_month_return": float(rolling_3m.min()) if not rolling_3m.empty else float("nan"),
                "covid_crash_return_2020_02_19_to_2020_03_23": period_return(
                    returns,
                    "2020-02-19",
                    "2020-03-23",
                ),
                "inflation_rate_shock_return_2022": period_return(
                    returns,
                    "2022-01-01",
                    "2022-12-31",
                ),
            }
        )

    return pd.DataFrame(rows)
