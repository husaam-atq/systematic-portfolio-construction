from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class SimulationResult:
    performance: pd.DataFrame
    pre_trade_weights: pd.DataFrame
    post_trade_weights: pd.DataFrame
    asset_return_contributions: pd.DataFrame
    target_weights: pd.DataFrame


def slice_simulation(
    simulation: SimulationResult,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp | None = None,
) -> SimulationResult:
    index = simulation.performance.loc[start:end].index
    return SimulationResult(
        performance=simulation.performance.loc[index].copy(),
        pre_trade_weights=simulation.pre_trade_weights.loc[index].copy(),
        post_trade_weights=simulation.post_trade_weights.loc[index].copy(),
        asset_return_contributions=simulation.asset_return_contributions.loc[index].copy(),
        target_weights=simulation.target_weights.loc[start:end].copy(),
    )


def common_start_targets(
    simulations: dict[str, SimulationResult],
    asset_columns: pd.Index | list[str],
    start: pd.Timestamp,
) -> dict[str, pd.DataFrame]:
    """Create restart targets so compared strategies share one inception date."""
    columns = list(asset_columns)
    restarted: dict[str, pd.DataFrame] = {}
    for strategy, simulation in simulations.items():
        if start not in simulation.post_trade_weights.index:
            raise ValueError(f"Common start {start} is unavailable for {strategy}.")
        initial = simulation.post_trade_weights.loc[[start], columns]
        future = simulation.target_weights.loc[simulation.target_weights.index > start, columns]
        restarted[strategy] = pd.concat([initial, future]).sort_index()
    return restarted


def simulate_portfolio(
    asset_returns: pd.DataFrame,
    target_weights: pd.DataFrame,
    cash_returns: pd.Series,
    transaction_cost_bps: float = 5.0,
    financing_spread_annual: float = 0.005,
    max_leverage: float = 1.5,
) -> SimulationResult:
    """
    Simulate close-to-close returns with trades after each date's realised return.

    A target indexed by date t is formed and traded at the close of t. It does
    not earn asset return t; it first earns return t+1. Between target dates,
    holdings drift with realised asset and cash returns.
    """
    if not asset_returns.index.equals(cash_returns.reindex(asset_returns.index).index):
        raise ValueError("Cash returns could not be aligned to asset-return dates.")
    if asset_returns.isna().any().any():
        raise ValueError("Asset returns contain missing values.")
    if not asset_returns.index.is_monotonic_increasing or asset_returns.index.has_duplicates:
        raise ValueError("Asset-return dates must be unique and increasing.")

    targets = target_weights.reindex(columns=asset_returns.columns).sort_index()
    if not targets.index.isin(asset_returns.index).all():
        raise ValueError("Every target date must be an asset-return date.")
    target_values = targets.dropna(how="all").fillna(0.0)
    if (target_values < -1e-12).any().any():
        raise ValueError("Only long asset weights are supported.")
    exposures = target_values.sum(axis=1)
    if (exposures > max_leverage + 1e-10).any():
        raise ValueError("A target exceeds the leverage cap.")

    columns = list(asset_returns.columns)
    pre_trade = pd.DataFrame(index=asset_returns.index, columns=[*columns, "CASH"], dtype=float)
    post_trade = pre_trade.copy()
    contributions = pd.DataFrame(index=asset_returns.index, columns=[*columns, "CASH"], dtype=float)
    records: list[dict[str, float]] = []

    asset_weights = pd.Series(0.0, index=columns)
    cash_weight = 1.0
    active = False
    cost_rate_per_turnover = transaction_cost_bps / 10000.0
    financing_spread_daily = financing_spread_annual / 252.0
    aligned_cash_returns = cash_returns.reindex(asset_returns.index).fillna(0.0)

    for date, returns_today in asset_returns.iterrows():
        cash_rate = float(aligned_cash_returns.loc[date])
        effective_cash_rate = cash_rate if cash_weight >= 0.0 else cash_rate + financing_spread_daily

        asset_contribution = asset_weights * returns_today
        cash_contribution = cash_weight * effective_cash_rate
        gross_return = float(asset_contribution.sum() + cash_contribution)
        gross_factor = 1.0 + gross_return
        if gross_factor <= 0.0:
            raise ValueError(f"Portfolio value became non-positive on {date.date()}.")

        drifted_assets = asset_weights * (1.0 + returns_today) / gross_factor
        drifted_cash = cash_weight * (1.0 + effective_cash_rate) / gross_factor
        pre_trade.loc[date, columns] = drifted_assets
        pre_trade.loc[date, "CASH"] = drifted_cash
        contributions.loc[date, columns] = asset_contribution
        contributions.loc[date, "CASH"] = cash_contribution

        turnover = 0.0
        transaction_cost_rate = 0.0
        if date in target_values.index:
            target = target_values.loc[date]
            target_exposure = float(target.sum())
            target_cash = 1.0 - target_exposure
            turnover = float((target - drifted_assets).abs().sum())
            transaction_cost_rate = turnover * cost_rate_per_turnover
            asset_weights = target.copy()
            cash_weight = target_cash
            active = True
        else:
            asset_weights = drifted_assets
            cash_weight = float(drifted_cash)

        net_factor = gross_factor * (1.0 - transaction_cost_rate)
        net_return = net_factor - 1.0
        post_trade.loc[date, columns] = asset_weights
        post_trade.loc[date, "CASH"] = cash_weight
        records.append(
            {
                "gross_return": gross_return,
                "turnover": turnover,
                "transaction_cost_rate": transaction_cost_rate,
                "transaction_cost_return": gross_factor * transaction_cost_rate,
                "net_return": net_return,
                "gross_exposure": float(asset_weights.abs().sum()),
                "cash_weight": cash_weight,
                "financing_weight": min(cash_weight, 0.0),
                "active": float(active),
            }
        )

    performance = pd.DataFrame(records, index=asset_returns.index)
    inactive = performance["active"] == 0.0
    inactive_fields = [column for column in performance.columns if column != "active"]
    performance.loc[inactive, inactive_fields] = np.nan
    contributions.loc[inactive, :] = np.nan
    return SimulationResult(performance, pre_trade, post_trade, contributions, target_values)


def delay_targets(
    targets: pd.DataFrame,
    trading_dates: pd.DatetimeIndex,
    extra_delay_days: int,
) -> pd.DataFrame:
    """Move decision/trade dates forward by a fixed number of trading sessions."""
    if extra_delay_days < 0:
        raise ValueError("Delay cannot be negative.")
    if extra_delay_days == 0:
        return targets.copy()
    rows = []
    dates = []
    for date, row in targets.dropna(how="all").iterrows():
        location = trading_dates.get_indexer([date])[0]
        delayed_location = location + extra_delay_days
        if location >= 0 and delayed_location < len(trading_dates):
            dates.append(trading_dates[delayed_location])
            rows.append(row.to_numpy())
    return pd.DataFrame(rows, index=pd.DatetimeIndex(dates), columns=targets.columns)
