from __future__ import annotations
import numpy as np
import pandas as pd

def zscore_cs(x: pd.Series) -> pd.Series:
    mu = x.mean(skipna=True)
    sd = x.std(skipna=True, ddof=0)
    if sd == 0 or np.isnan(sd):
        return x * 0.0
    return (x - mu) / sd

def momentum_12_1(prices: pd.DataFrame) -> pd.DataFrame:
    r_12m = prices.pct_change(252)
    r_1m  = prices.pct_change(21)
    return r_12m - r_1m

def low_vol(prices: pd.DataFrame, window: int = 63) -> pd.DataFrame:
    vol = prices.pct_change().rolling(window).std()
    return -vol

def reversal_1w(prices: pd.DataFrame) -> pd.DataFrame:
    return -prices.pct_change(5)

def to_signal(raw: pd.DataFrame) -> pd.DataFrame:
    return raw.apply(zscore_cs, axis=1)
