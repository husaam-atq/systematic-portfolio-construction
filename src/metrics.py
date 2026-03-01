from __future__ import annotations
import numpy as np
import pandas as pd

def ann_return(r: pd.Series, ppy: int = 252) -> float:
    r = r.dropna()
    if r.empty: return float("nan")
    return (1+r).prod() ** (ppy/len(r)) - 1

def ann_vol(r: pd.Series, ppy: int = 252) -> float:
    r = r.dropna()
    if r.empty: return float("nan")
    return r.std(ddof=0) * np.sqrt(ppy)

def sharpe(r: pd.Series, ppy: int = 252) -> float:
    r = r.dropna()
    if r.empty: return float("nan")
    v = r.std(ddof=0)
    if v == 0: return float("nan")
    return r.mean() / v * np.sqrt(ppy)

def max_dd(r: pd.Series) -> float:
    r = r.dropna()
    if r.empty: return float("nan")
    eq = (1+r).cumprod()
    peak = eq.cummax()
    dd = eq/peak - 1
    return dd.min()

def summary(r: pd.Series) -> pd.Series:
    return pd.Series({
        "Ann.Return": ann_return(r),
        "Ann.Vol": ann_vol(r),
        "Sharpe": sharpe(r),
        "MaxDD": max_dd(r),
        "Obs": int(r.dropna().shape[0])
    })
