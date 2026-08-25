from __future__ import annotations

import pandas as pd


def trend_eligibility(
    prices: pd.DataFrame,
    moving_average_days: int = 200,
    cash_proxy: str = "SHY",
) -> pd.DataFrame:
    """Signal observed at close t and eligible for returns beginning after close t."""
    moving_average = prices.rolling(moving_average_days, min_periods=moving_average_days).mean()
    eligible = prices > moving_average
    if cash_proxy in eligible.columns:
        eligible[cash_proxy] = True
    return eligible.fillna(False)


def dual_momentum(
    prices: pd.DataFrame,
    lookback_days: int = 252,
    skip_days: int = 21,
) -> pd.DataFrame:
    """Twelve-month momentum excluding the most recent month, known at close t."""
    if lookback_days <= skip_days:
        raise ValueError("Momentum lookback must exceed the skip period.")
    return prices.shift(skip_days) / prices.shift(lookback_days) - 1.0
