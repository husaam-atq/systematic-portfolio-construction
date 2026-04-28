from __future__ import annotations

import numpy as np
import pandas as pd


def component_risk_contribution(
    weights: pd.Series,
    covariance: pd.DataFrame,
    periods_per_year: int = 252,
) -> tuple[pd.Series, float]:
    """Return percentage component risk contributions and annual volatility."""
    aligned_cov = covariance.loc[weights.index, weights.index].fillna(0.0)
    weights_array = weights.to_numpy(dtype=float)
    covariance_array = aligned_cov.to_numpy(dtype=float)

    portfolio_variance = float(weights_array @ covariance_array @ weights_array)
    if portfolio_variance <= 1e-16:
        contributions = pd.Series(np.nan, index=weights.index)
        return contributions, float("nan")

    marginal = covariance_array @ weights_array
    contributions = weights_array * marginal / portfolio_variance
    portfolio_volatility = np.sqrt(portfolio_variance * periods_per_year)
    return pd.Series(contributions, index=weights.index), float(portfolio_volatility)


def summarize_risk_contributions(risk_records: pd.DataFrame) -> pd.DataFrame:
    if risk_records.empty:
        return pd.DataFrame()

    grouped = (
        risk_records.groupby(["strategy", "asset"], as_index=False)
        .agg(
            average_component_risk_contribution=("component_risk_contribution", "mean"),
            average_portfolio_volatility=("portfolio_volatility", "mean"),
        )
    )

    concentration_base = grouped.copy()
    concentration_base["absolute_contribution"] = concentration_base[
        "average_component_risk_contribution"
    ].abs()
    contribution_sum = concentration_base.groupby("strategy")["absolute_contribution"].transform("sum")
    concentration_base["normalized_absolute_contribution"] = (
        concentration_base["absolute_contribution"] / contribution_sum
    )
    concentration = (
        concentration_base.assign(square=lambda frame: frame["normalized_absolute_contribution"] ** 2)
        .groupby("strategy")
        .agg(risk_concentration=("square", "sum"))
    )
    concentration["effective_number_of_risk_contributors"] = 1.0 / concentration["risk_concentration"]

    return grouped.merge(concentration, on="strategy", how="left")


def summarize_average_weights(weights: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for strategy, frame in weights.items():
        active = frame[frame.sum(axis=1) > 0.0]
        averages = active.mean(axis=0) if not active.empty else frame.mean(axis=0)
        for asset, value in averages.items():
            rows.append({"strategy": strategy, "asset": asset, "average_weight": value})
    return pd.DataFrame(rows)
