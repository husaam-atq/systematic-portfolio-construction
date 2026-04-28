from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize


STRATEGY_NAMES = [
    "Equal Weight",
    "Inverse Volatility",
    "Minimum Variance",
    "Maximum Sharpe",
    "Risk Parity",
]


def _equal_weight(n_assets: int) -> np.ndarray:
    return np.repeat(1.0 / n_assets, n_assets)


def _clean_covariance(returns: pd.DataFrame) -> np.ndarray:
    cov = returns.cov().to_numpy(dtype=float)
    cov = np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)
    cov = cov + np.eye(cov.shape[0]) * 1e-10
    return cov


def _normalize(weights: np.ndarray) -> np.ndarray:
    weights = np.clip(np.asarray(weights, dtype=float), 0.0, None)
    total = weights.sum()
    if total <= 0:
        return _equal_weight(len(weights))
    return weights / total


def cap_weights(weights: np.ndarray, max_weight: float = 0.40) -> np.ndarray:
    """Project long-only weights to a simple max-weight cap and renormalize."""
    weights = _normalize(weights)
    n_assets = len(weights)
    if max_weight * n_assets < 1.0:
        raise ValueError("Max weight cap is infeasible for the asset count.")

    capped = np.zeros(n_assets)
    remaining = np.arange(n_assets)
    remaining_budget = 1.0
    working = weights.copy()

    while len(remaining) > 0:
        scaled = _normalize(working[remaining]) * remaining_budget
        over_cap = scaled > max_weight
        if not over_cap.any():
            capped[remaining] = scaled
            break

        capped_assets = remaining[over_cap]
        capped[capped_assets] = max_weight
        remaining_budget -= max_weight * len(capped_assets)
        remaining = remaining[~over_cap]

        if remaining_budget <= 1e-12:
            break

    return _normalize(capped)


def equal_weight(returns: pd.DataFrame, max_weight: float = 0.40) -> pd.Series:
    del max_weight
    return pd.Series(_equal_weight(returns.shape[1]), index=returns.columns)


def inverse_volatility(returns: pd.DataFrame, max_weight: float = 0.40) -> pd.Series:
    vol = returns.std(ddof=0).replace(0.0, np.nan)
    inv_vol = (1.0 / vol).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    weights = cap_weights(inv_vol.to_numpy(dtype=float), max_weight=max_weight)
    return pd.Series(weights, index=returns.columns)


def minimum_variance(returns: pd.DataFrame, max_weight: float = 0.40) -> pd.Series:
    cov = _clean_covariance(returns)
    n_assets = returns.shape[1]
    x0 = inverse_volatility(returns, max_weight=max_weight).to_numpy()

    def objective(weights: np.ndarray) -> float:
        return float(weights @ cov @ weights)

    result = minimize(
        objective,
        x0=x0,
        method="SLSQP",
        bounds=[(0.0, max_weight)] * n_assets,
        constraints={"type": "eq", "fun": lambda weights: weights.sum() - 1.0},
        options={"maxiter": 500, "ftol": 1e-12},
    )
    weights = result.x if result.success else x0
    return pd.Series(cap_weights(weights, max_weight=max_weight), index=returns.columns)


def maximum_sharpe(returns: pd.DataFrame, max_weight: float = 0.40) -> pd.Series:
    cov = _clean_covariance(returns) * 252.0
    mean_returns = returns.mean().to_numpy(dtype=float) * 252.0
    n_assets = returns.shape[1]
    x0 = equal_weight(returns).to_numpy()

    def objective(weights: np.ndarray) -> float:
        volatility = np.sqrt(max(float(weights @ cov @ weights), 0.0))
        if volatility <= 1e-12:
            return 1e6
        return -float(weights @ mean_returns) / volatility

    result = minimize(
        objective,
        x0=x0,
        method="SLSQP",
        bounds=[(0.0, max_weight)] * n_assets,
        constraints={"type": "eq", "fun": lambda weights: weights.sum() - 1.0},
        options={"maxiter": 500, "ftol": 1e-12},
    )
    weights = result.x if result.success else x0
    return pd.Series(cap_weights(weights, max_weight=max_weight), index=returns.columns)


def risk_parity(returns: pd.DataFrame, max_weight: float = 0.40) -> pd.Series:
    cov = _clean_covariance(returns)
    n_assets = returns.shape[1]
    target = np.repeat(1.0 / n_assets, n_assets)
    x0 = inverse_volatility(returns, max_weight=max_weight).to_numpy()

    def objective(weights: np.ndarray) -> float:
        portfolio_variance = float(weights @ cov @ weights)
        if portfolio_variance <= 1e-16:
            return 1e6
        contribution = weights * (cov @ weights) / portfolio_variance
        return float(((contribution - target) ** 2).sum())

    result = minimize(
        objective,
        x0=x0,
        method="SLSQP",
        bounds=[(0.0, max_weight)] * n_assets,
        constraints={"type": "eq", "fun": lambda weights: weights.sum() - 1.0},
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    weights = result.x if result.success else x0
    return pd.Series(cap_weights(weights, max_weight=max_weight), index=returns.columns)


def estimate_allocation_weights(
    trailing_returns: pd.DataFrame,
    max_weight: float = 0.40,
) -> dict[str, pd.Series]:
    """Estimate all allocation methods from the same trailing return window."""
    clean_returns = trailing_returns.dropna(how="any")
    return {
        "Equal Weight": equal_weight(clean_returns, max_weight=max_weight),
        "Inverse Volatility": inverse_volatility(clean_returns, max_weight=max_weight),
        "Minimum Variance": minimum_variance(clean_returns, max_weight=max_weight),
        "Maximum Sharpe": maximum_sharpe(clean_returns, max_weight=max_weight),
        "Risk Parity": risk_parity(clean_returns, max_weight=max_weight),
    }
