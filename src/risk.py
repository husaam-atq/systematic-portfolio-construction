from __future__ import annotations

import numpy as np
import pandas as pd

from src.allocation import percentage_risk_contributions
from src.simulation import SimulationResult


def risk_snapshot(
    weights: pd.Series,
    covariance: pd.DataFrame,
    periods_per_year: int = 252,
) -> pd.DataFrame:
    covariance = covariance.loc[weights.index, weights.index]
    vector = weights.to_numpy()
    covariance_array = covariance.to_numpy()
    variance = float(vector @ covariance_array @ vector)
    portfolio_volatility = np.sqrt(max(variance, 0.0) * periods_per_year)
    percentage = percentage_risk_contributions(weights, covariance)
    volatility_contribution = percentage * portfolio_volatility
    absolute = percentage.abs()
    normalized_absolute = absolute / absolute.sum() if absolute.sum() > 0 else absolute
    risk_hhi = float((normalized_absolute**2).sum()) if not normalized_absolute.empty else np.nan
    gross_exposure = float(weights.abs().sum())
    normalized_weights = weights / gross_exposure if gross_exposure > 0 else weights
    weight_hhi = float((normalized_weights**2).sum())

    return pd.DataFrame(
        {
            "asset": weights.index,
            "weight": weights.to_numpy(),
            "volatility_contribution": volatility_contribution.to_numpy(),
            "percentage_risk_contribution": percentage.to_numpy(),
            "portfolio_volatility": portfolio_volatility,
            "weight_hhi": weight_hhi,
            "effective_assets_by_weight": 1.0 / weight_hhi if weight_hhi > 0 else np.nan,
            "risk_concentration_hhi": risk_hhi,
            "effective_assets_by_risk": 1.0 / risk_hhi if risk_hhi > 0 else np.nan,
            "maximum_asset_weight": float(normalized_weights.max()),
            "gross_exposure": gross_exposure,
        }
    )


def summarize_risk_records(records: pd.DataFrame) -> pd.DataFrame:
    if records.empty:
        return records
    asset_summary = (
        records.groupby(["strategy", "asset"], as_index=False)
        .agg(
            average_weight=("weight", "mean"),
            average_volatility_contribution=("volatility_contribution", "mean"),
            average_percentage_risk_contribution=("percentage_risk_contribution", "mean"),
        )
    )
    strategy_summary = (
        records.groupby("strategy", as_index=False)
        .agg(
            average_portfolio_volatility=("portfolio_volatility", "mean"),
            average_weight_hhi=("weight_hhi", "mean"),
            effective_assets_by_weight=("effective_assets_by_weight", "mean"),
            average_risk_concentration_hhi=("risk_concentration_hhi", "mean"),
            effective_assets_by_risk=("effective_assets_by_risk", "mean"),
            average_maximum_asset_weight=("maximum_asset_weight", "mean"),
            average_gross_exposure=("gross_exposure", "mean"),
        )
    )
    return asset_summary.merge(strategy_summary, on="strategy", how="left")


def return_contribution_summary(simulations: dict[str, SimulationResult]) -> pd.DataFrame:
    rows = []
    for strategy, simulation in simulations.items():
        active = simulation.asset_return_contributions.dropna(how="all")
        for asset in active.columns:
            contribution = active[asset].sum()
            rows.append(
                {
                    "strategy": strategy,
                    "asset": asset,
                    "arithmetic_return_contribution": float(contribution),
                    "annualized_arithmetic_contribution": float(contribution * 252.0 / len(active))
                    if len(active)
                    else np.nan,
                }
            )
    return pd.DataFrame(rows)


def drawdown_contribution_summary(simulations: dict[str, SimulationResult]) -> pd.DataFrame:
    """Additive contribution approximation over each strategy's maximum-drawdown window."""
    rows = []
    for strategy, simulation in simulations.items():
        returns = simulation.performance["net_return"].dropna()
        if returns.empty:
            continue
        equity = (1.0 + returns).cumprod()
        drawdown = equity / equity.cummax() - 1.0
        trough = drawdown.idxmin()
        peak = equity.loc[:trough].idxmax()
        contributions = simulation.asset_return_contributions.loc[peak:trough].sum()
        for asset, value in contributions.items():
            rows.append(
                {
                    "strategy": strategy,
                    "asset": asset,
                    "drawdown_peak": peak,
                    "drawdown_trough": trough,
                    "additive_drawdown_contribution": float(value),
                }
            )
    return pd.DataFrame(rows)


def risk_parity_verification(records: pd.DataFrame) -> pd.DataFrame:
    canonical = {
        "Risk Parity",
        "Risk Parity Shrinkage",
        "Risk Parity Vol Target 10%",
    }
    selected = records[records["strategy"].isin(canonical)].copy()
    if selected.empty:
        return pd.DataFrame()
    selected["target_risk_contribution"] = selected.groupby(["strategy", "date"])["asset"].transform(
        lambda series: 1.0 / len(series)
    )
    selected["absolute_error"] = (
        selected["percentage_risk_contribution"] - selected["target_risk_contribution"]
    ).abs()
    return (
        selected.groupby("strategy", as_index=False)
        .agg(
            mean_absolute_risk_contribution_error=("absolute_error", "mean"),
            maximum_risk_contribution_error=("absolute_error", "max"),
            observations=("date", "nunique"),
        )
    )


def summarize_average_weights(weights: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for strategy, frame in weights.items():
        active = frame.dropna(how="all")
        averages = active.mean() if not active.empty else frame.mean()
        for asset, value in averages.items():
            rows.append({"strategy": strategy, "asset": asset, "average_weight": value})
    return pd.DataFrame(rows)
