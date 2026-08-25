from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.optimize import minimize
from scipy.spatial.distance import squareform


@dataclass
class AllocationResult:
    weights: pd.Series
    covariance: pd.DataFrame
    expected_returns: pd.Series
    diagnostics: dict[str, float | int | bool | str]


def covariance_matrix(returns: pd.DataFrame, shrinkage: float = 0.0) -> pd.DataFrame:
    sample = returns.cov().astype(float)
    diagonal = pd.DataFrame(np.diag(np.diag(sample)), index=sample.index, columns=sample.columns)
    intensity = float(np.clip(shrinkage, 0.0, 1.0))
    covariance = (1.0 - intensity) * sample + intensity * diagonal
    covariance += np.eye(len(covariance)) * 1e-10
    return covariance


def estimate_expected_returns(returns: pd.DataFrame, estimator: str = "sample") -> pd.Series:
    if estimator == "sample":
        estimate = returns.mean()
    elif estimator == "ewma":
        estimate = returns.ewm(halflife=126, adjust=False).mean().iloc[-1]
    elif estimator == "grand_mean_shrink":
        sample = returns.mean()
        estimate = 0.5 * sample + 0.5 * sample.mean()
    else:
        raise ValueError(f"Unknown expected-return estimator: {estimator}")
    return estimate.astype(float) * 252.0


def project_capped_simplex(weights: np.ndarray, max_weight: float) -> np.ndarray:
    weights = np.clip(np.asarray(weights, dtype=float), 0.0, None)
    n_assets = len(weights)
    if max_weight * n_assets < 1.0 - 1e-12:
        raise ValueError("Maximum-weight constraint is infeasible.")
    if weights.sum() <= 0:
        weights = np.ones(n_assets)
    weights /= weights.sum()

    fixed = np.zeros(n_assets, dtype=bool)
    result = np.zeros(n_assets)
    remaining_budget = 1.0
    while (~fixed).any():
        available = ~fixed
        base = weights[available]
        if base.sum() <= 0:
            candidate = np.repeat(remaining_budget / available.sum(), available.sum())
        else:
            candidate = base / base.sum() * remaining_budget
        over = candidate > max_weight + 1e-12
        available_indices = np.flatnonzero(available)
        if not over.any():
            result[available_indices] = candidate
            break
        capped_indices = available_indices[over]
        result[capped_indices] = max_weight
        fixed[capped_indices] = True
        remaining_budget = 1.0 - result.sum()
    result /= result.sum()
    return result


def equal_weight(returns: pd.DataFrame, max_weight: float = 1.0) -> pd.Series:
    weights = np.repeat(1.0 / returns.shape[1], returns.shape[1])
    return pd.Series(project_capped_simplex(weights, max_weight), index=returns.columns)


def inverse_volatility(returns: pd.DataFrame, max_weight: float = 0.40) -> pd.Series:
    volatility = returns.std(ddof=0).replace(0.0, np.nan)
    raw = (1.0 / volatility).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy()
    return pd.Series(project_capped_simplex(raw, max_weight), index=returns.columns)


def _optimizer_result(
    raw_weights: np.ndarray,
    returns: pd.DataFrame,
    covariance: pd.DataFrame,
    expected_returns: pd.Series,
    max_weight: float,
    success: bool,
    status: str,
    method: str,
) -> AllocationResult:
    weights = pd.Series(project_capped_simplex(raw_weights, max_weight), index=returns.columns)
    condition_number = float(np.linalg.cond(covariance.to_numpy()))
    hhi = float((weights**2).sum())
    diagnostics: dict[str, float | int | bool | str] = {
        "method": method,
        "solver_success": bool(success),
        "solver_status": status,
        "condition_number": condition_number,
        "bound_count": int(np.isclose(weights, max_weight, atol=1e-5).sum()),
        "max_weight": float(weights.max()),
        "weight_hhi": hhi,
        "effective_assets": float(1.0 / hhi),
        "weight_dispersion": float(weights.std(ddof=0)),
    }
    return AllocationResult(weights, covariance, expected_returns, diagnostics)


def minimum_variance(
    returns: pd.DataFrame,
    max_weight: float = 0.40,
    shrinkage: float = 0.0,
) -> AllocationResult:
    covariance = covariance_matrix(returns, shrinkage=shrinkage)
    expected = estimate_expected_returns(returns)
    initial = inverse_volatility(returns, max_weight=max_weight).to_numpy()
    cov_array = covariance.to_numpy()

    result = minimize(
        lambda weights: float(weights @ cov_array @ weights),
        x0=initial,
        method="SLSQP",
        bounds=[(0.0, max_weight)] * len(initial),
        constraints={"type": "eq", "fun": lambda weights: weights.sum() - 1.0},
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    raw = result.x if result.success else initial
    return _optimizer_result(
        raw,
        returns,
        covariance,
        expected,
        max_weight,
        result.success,
        result.message,
        "minimum_variance",
    )


def maximum_sharpe(
    returns: pd.DataFrame,
    max_weight: float = 0.40,
    shrinkage: float = 0.0,
    expected_return_estimator: str = "sample",
    risk_free_asset: str | None = "SHY",
) -> AllocationResult:
    covariance = covariance_matrix(returns, shrinkage=shrinkage) * 252.0
    expected = estimate_expected_returns(returns, estimator=expected_return_estimator)
    risk_free_rate = float(expected.get(risk_free_asset, 0.0)) if risk_free_asset else 0.0
    expected_excess = expected - risk_free_rate
    initial = equal_weight(returns, max_weight=max_weight).to_numpy()
    cov_array = covariance.to_numpy()
    mean_array = expected_excess.to_numpy()

    def objective(weights: np.ndarray) -> float:
        volatility = np.sqrt(max(float(weights @ cov_array @ weights), 1e-16))
        return -float(weights @ mean_array) / volatility

    result = minimize(
        objective,
        x0=initial,
        method="SLSQP",
        bounds=[(0.0, max_weight)] * len(initial),
        constraints={"type": "eq", "fun": lambda weights: weights.sum() - 1.0},
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    raw = result.x if result.success else initial
    allocation = _optimizer_result(
        raw,
        returns,
        covariance / 252.0,
        expected,
        max_weight,
        result.success,
        result.message,
        f"maximum_sharpe_{expected_return_estimator}",
    )
    allocation.diagnostics["estimated_sharpe"] = float(-objective(allocation.weights.to_numpy()))
    allocation.diagnostics["risk_free_asset"] = risk_free_asset or "zero_hurdle"
    allocation.diagnostics["estimated_annual_risk_free_rate"] = risk_free_rate
    return allocation


def percentage_risk_contributions(weights: pd.Series, covariance: pd.DataFrame) -> pd.Series:
    covariance = covariance.loc[weights.index, weights.index]
    vector = weights.to_numpy()
    variance = float(vector @ covariance.to_numpy() @ vector)
    if variance <= 1e-16:
        return pd.Series(np.nan, index=weights.index)
    return pd.Series(vector * (covariance.to_numpy() @ vector) / variance, index=weights.index)


def risk_parity(
    returns: pd.DataFrame,
    max_weight: float = 0.40,
    shrinkage: float = 0.0,
) -> AllocationResult:
    covariance = covariance_matrix(returns, shrinkage=shrinkage)
    expected = estimate_expected_returns(returns)
    initial = inverse_volatility(returns, max_weight=max_weight).to_numpy()
    target = np.repeat(1.0 / len(initial), len(initial))

    def objective(weights: np.ndarray) -> float:
        series = pd.Series(weights, index=returns.columns)
        contributions = percentage_risk_contributions(series, covariance).to_numpy()
        if np.isnan(contributions).any():
            return 1e6
        return float(((contributions - target) ** 2).sum())

    result = minimize(
        objective,
        x0=initial,
        method="SLSQP",
        bounds=[(0.0, max_weight)] * len(initial),
        constraints={"type": "eq", "fun": lambda weights: weights.sum() - 1.0},
        options={"maxiter": 2000, "ftol": 1e-14},
    )
    raw = result.x if result.success else initial
    allocation = _optimizer_result(
        raw,
        returns,
        covariance,
        expected,
        max_weight,
        result.success,
        result.message,
        "risk_parity",
    )
    contributions = percentage_risk_contributions(allocation.weights, covariance)
    allocation.diagnostics["max_risk_contribution_error"] = float(
        (contributions - 1.0 / len(contributions)).abs().max()
    )
    return allocation


def _cluster_variance(covariance: pd.DataFrame, assets: list[str]) -> float:
    sub_covariance = covariance.loc[assets, assets]
    diagonal = np.diag(sub_covariance)
    inverse = np.divide(1.0, diagonal, out=np.zeros_like(diagonal), where=diagonal > 0)
    weights = inverse / inverse.sum() if inverse.sum() > 0 else np.repeat(1.0 / len(assets), len(assets))
    return float(weights @ sub_covariance.to_numpy() @ weights)


def hierarchical_risk_parity(
    returns: pd.DataFrame,
    max_weight: float = 0.40,
    shrinkage: float = 0.0,
) -> AllocationResult:
    covariance = covariance_matrix(returns, shrinkage=shrinkage)
    expected = estimate_expected_returns(returns)
    standard_deviation = np.sqrt(np.diag(covariance))
    denominator = np.outer(standard_deviation, standard_deviation)
    correlation = np.divide(
        covariance.to_numpy(),
        denominator,
        out=np.eye(len(covariance)),
        where=denominator > 0,
    )
    correlation = np.clip(correlation, -1.0, 1.0)
    distance = np.sqrt(np.maximum((1.0 - correlation) / 2.0, 0.0))
    ordering = leaves_list(linkage(squareform(distance, checks=False), method="single"))
    ordered_assets = list(returns.columns[ordering])

    weights = pd.Series(1.0, index=ordered_assets)
    clusters = [ordered_assets]
    while clusters:
        next_clusters: list[list[str]] = []
        for cluster in clusters:
            if len(cluster) <= 1:
                continue
            split = len(cluster) // 2
            left, right = cluster[:split], cluster[split:]
            left_variance = _cluster_variance(covariance, left)
            right_variance = _cluster_variance(covariance, right)
            total = left_variance + right_variance
            left_allocation = right_variance / total if total > 0 else 0.5
            weights.loc[left] *= left_allocation
            weights.loc[right] *= 1.0 - left_allocation
            next_clusters.extend([left, right])
        clusters = next_clusters

    weights = pd.Series(
        project_capped_simplex(weights.reindex(returns.columns).to_numpy(), max_weight),
        index=returns.columns,
    )
    condition_number = float(np.linalg.cond(covariance.to_numpy()))
    hhi = float((weights**2).sum())
    diagnostics: dict[str, float | int | bool | str] = {
        "method": "hierarchical_risk_parity",
        "solver_success": True,
        "solver_status": "hierarchical allocation",
        "condition_number": condition_number,
        "bound_count": int(np.isclose(weights, max_weight, atol=1e-5).sum()),
        "max_weight": float(weights.max()),
        "weight_hhi": hhi,
        "effective_assets": float(1.0 / hhi),
        "weight_dispersion": float(weights.std(ddof=0)),
    }
    return AllocationResult(weights, covariance, expected, diagnostics)


def fixed_allocation(returns: pd.DataFrame, weights: dict[str, float], method: str) -> AllocationResult:
    series = pd.Series(0.0, index=returns.columns)
    for asset, weight in weights.items():
        series.loc[asset] = weight
    covariance = covariance_matrix(returns)
    hhi = float((series**2).sum())
    diagnostics: dict[str, float | int | bool | str] = {
        "method": method,
        "solver_success": True,
        "solver_status": "fixed allocation",
        "condition_number": float(np.linalg.cond(covariance.to_numpy())),
        "bound_count": 0,
        "max_weight": float(series.max()),
        "weight_hhi": hhi,
        "effective_assets": float(1.0 / hhi),
        "weight_dispersion": float(series.std(ddof=0)),
    }
    return AllocationResult(series, covariance, estimate_expected_returns(returns), diagnostics)


def estimate_method(
    method: str,
    trailing_returns: pd.DataFrame,
    max_weight: float = 0.40,
    shrinkage: float = 0.25,
    expected_return_estimator: str = "sample",
    risk_free_asset: str | None = "SHY",
) -> AllocationResult:
    if method == "Equal Weight":
        return fixed_allocation(
            trailing_returns,
            {asset: 1.0 / trailing_returns.shape[1] for asset in trailing_returns.columns},
            "equal_weight",
        )
    if method == "Traditional 60/40":
        return fixed_allocation(trailing_returns, {"SPY": 0.60, "IEF": 0.40}, "traditional_60_40")
    if method == "Inverse Volatility":
        weights = inverse_volatility(trailing_returns, max_weight=max_weight)
        covariance = covariance_matrix(trailing_returns)
        return _optimizer_result(
            weights.to_numpy(),
            trailing_returns,
            covariance,
            estimate_expected_returns(trailing_returns),
            max_weight,
            True,
            "closed form",
            "inverse_volatility",
        )
    if method == "Minimum Variance":
        return minimum_variance(trailing_returns, max_weight=max_weight)
    if method == "Minimum Variance Shrinkage":
        return minimum_variance(trailing_returns, max_weight=max_weight, shrinkage=shrinkage)
    if method == "Maximum Sharpe":
        return maximum_sharpe(
            trailing_returns,
            max_weight=max_weight,
            expected_return_estimator=expected_return_estimator,
            risk_free_asset=risk_free_asset,
        )
    if method == "Maximum Sharpe Shrinkage":
        return maximum_sharpe(
            trailing_returns,
            max_weight=max_weight,
            shrinkage=shrinkage,
            expected_return_estimator=expected_return_estimator,
            risk_free_asset=risk_free_asset,
        )
    if method == "Risk Parity":
        return risk_parity(trailing_returns, max_weight=max_weight)
    if method == "Risk Parity Shrinkage":
        return risk_parity(trailing_returns, max_weight=max_weight, shrinkage=shrinkage)
    if method == "Hierarchical Risk Parity":
        return hierarchical_risk_parity(trailing_returns, max_weight=max_weight)
    raise ValueError(f"Unknown allocation method: {method}")
