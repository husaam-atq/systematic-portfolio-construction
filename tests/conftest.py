from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_prices() -> pd.DataFrame:
    random = np.random.default_rng(42)
    dates = pd.bdate_range("2018-01-01", periods=700)
    tickers = ["SPY", "EFA", "EEM", "TLT", "IEF", "SHY", "GLD", "VNQ", "DBC"]
    means = np.array([0.00035, 0.00025, 0.00020, 0.00010, 0.00008, 0.00003, 0.00015, 0.00025, 0.00012])
    volatility = np.array([0.010, 0.011, 0.013, 0.008, 0.004, 0.001, 0.009, 0.012, 0.011])
    common = random.normal(0.0, 0.004, size=(len(dates), 1))
    shocks = random.normal(0.0, 1.0, size=(len(dates), len(tickers))) * volatility
    returns = means + 0.35 * common + shocks
    prices = 100.0 * pd.DataFrame(1.0 + returns, index=dates, columns=tickers).cumprod()
    return prices
