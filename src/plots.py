from __future__ import annotations
import matplotlib.pyplot as plt
import pandas as pd

def equity_curve(r: pd.Series, title: str, path: str | None = None):
    rr = r.fillna(0.0)
    eq = (1+rr).cumprod()
    plt.figure()
    plt.plot(eq.index, eq.values)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Equity (Start=1.0)")
    plt.tight_layout()
    if path:
        plt.savefig(path, dpi=150)
    plt.show()
