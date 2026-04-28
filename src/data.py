from __future__ import annotations

import pandas as pd
import yfinance as yf


ETF_UNIVERSE: dict[str, str] = {
    "SPY": "US equities",
    "EFA": "Developed ex-US equities",
    "EEM": "Emerging markets equities",
    "TLT": "Long-term US Treasuries",
    "IEF": "Intermediate US Treasuries",
    "SHY": "Short-duration US Treasuries / cash proxy",
    "GLD": "Gold",
    "VNQ": "REITs",
    "DBC": "Commodities",
}


def download_prices(
    tickers: list[str],
    start: str = "2010-01-01",
    end: str | None = None,
) -> pd.DataFrame:
    """Download adjusted close prices from yfinance."""
    raw = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=False,
    )

    if raw.empty:
        raise ValueError("No price data returned by yfinance.")

    if isinstance(raw.columns, pd.MultiIndex):
        closes = []
        for ticker in tickers:
            if (ticker, "Close") in raw.columns:
                closes.append(raw[(ticker, "Close")].rename(ticker))
            elif ("Close", ticker) in raw.columns:
                closes.append(raw[("Close", ticker)].rename(ticker))
            else:
                raise KeyError(f"Missing adjusted close data for {ticker}.")
        prices = pd.concat(closes, axis=1)
    else:
        if len(tickers) != 1:
            raise ValueError("Expected multi-ticker yfinance output.")
        prices = raw["Close"].to_frame(name=tickers[0])

    prices = prices.sort_index().ffill().dropna(how="any")
    prices.index = pd.to_datetime(prices.index)
    return prices[tickers]


def calculate_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Convert adjusted close prices into daily simple returns."""
    return prices.pct_change().dropna(how="all")
