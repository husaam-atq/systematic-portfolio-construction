from __future__ import annotations
import pandas as pd

def combine_signals(signals: dict[str, pd.DataFrame], weights: dict[str, float]) -> pd.DataFrame:
    """Weighted sum of z-scored signals."""
    out = None
    for k, sig in signals.items():
        w = weights.get(k, 0.0)
        if out is None:
            out = w * sig
        else:
            out = out.add(w * sig, fill_value=0.0)
    return out

def long_short_weights(signal: pd.DataFrame, q_long: float = 0.8, q_short: float = 0.2) -> pd.DataFrame:
    """Equal-weight top/bottom quantile long/short, dollar-neutral."""
    w = pd.DataFrame(index=signal.index, columns=signal.columns, dtype=float)
    for dt, row in signal.iterrows():
        x = row.dropna()
        if len(x) < 10:
            continue
        a = x.quantile(q_long)
        b = x.quantile(q_short)
        longs = x[x >= a].index
        shorts = x[x <= b].index

        ww = pd.Series(0.0, index=row.index)
        if len(longs) > 0:
            ww.loc[longs] = 1.0 / len(longs)
        if len(shorts) > 0:
            ww.loc[shorts] = -1.0 / len(shorts)

        # normalize each side to 1
        pos = ww[ww > 0].sum()
        neg = -ww[ww < 0].sum()
        if pos > 0:
            ww[ww > 0] /= pos
        if neg > 0:
            ww[ww < 0] /= neg

        w.loc[dt] = ww
    return w
