from __future__ import annotations
import pandas as pd

def backtest(returns: pd.DataFrame, weights: pd.DataFrame, lag: int = 1, tc_bps: float = 5.0) -> pd.DataFrame:
    returns, weights = returns.align(weights, join="inner", axis=0)
    w = weights.shift(lag).fillna(0.0)

    gross = (w * returns).sum(axis=1)
    turnover = (w.diff().abs().sum(axis=1)).fillna(0.0)
    costs = (tc_bps / 10000.0) * turnover
    net = gross - costs

    return pd.DataFrame({"gross": gross, "turnover": turnover, "costs": costs, "net": net})
