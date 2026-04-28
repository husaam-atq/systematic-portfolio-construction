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
