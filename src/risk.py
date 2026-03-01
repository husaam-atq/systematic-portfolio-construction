from __future__ import annotations
import numpy as np
import pandas as pd

def vol_target_scale(
    weights: pd.DataFrame,
    returns: pd.DataFrame,
    target_ann_vol: float = 0.12,
    lookback: int = 63,
    periods_per_year: int = 252
) -> pd.DataFrame:
    """Scale portfolio exposure to hit target volatility using rolling realized vol."""
    # portfolio returns using same-day weights (we'll lag later in backtest)
    w = weights.fillna(0.0)
    pr = (w * returns).sum(axis=1)

    rv = pr.rolling(lookback).std() * np.sqrt(periods_per_year)
    scale = target_ann_vol / rv
    scale = scale.clip(lower=0.0, upper=5.0).fillna(0.0)  # cap leverage

    scaled = w.mul(scale, axis=0)
    return scaled
