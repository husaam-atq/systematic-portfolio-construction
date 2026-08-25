from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf

from src.config import ASSET_CLASS_MAP


ETF_UNIVERSE = ASSET_CLASS_MAP


def download_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Download auto-adjusted closes without requesting data beyond ``end``."""
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

    closes: list[pd.Series] = []
    if isinstance(raw.columns, pd.MultiIndex):
        for ticker in tickers:
            if (ticker, "Close") in raw.columns:
                closes.append(raw[(ticker, "Close")].rename(ticker))
            elif ("Close", ticker) in raw.columns:
                closes.append(raw[("Close", ticker)].rename(ticker))
            else:
                raise KeyError(f"Missing adjusted close data for {ticker}.")
        prices = pd.concat(closes, axis=1)
    elif len(tickers) == 1:
        prices = raw["Close"].to_frame(name=tickers[0])
    else:
        raise ValueError("Expected multi-ticker yfinance output.")

    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    return prices.sort_index()[tickers]


def clean_prices(prices: pd.DataFrame, max_forward_fill_days: int = 3) -> pd.DataFrame:
    """Fill only short internal gaps, then retain complete cross-asset observations."""
    if not prices.index.is_monotonic_increasing or prices.index.has_duplicates:
        raise ValueError("Price dates must be unique and increasing.")
    if (prices <= 0).any().any():
        raise ValueError("Prices must be strictly positive.")
    cleaned = prices.ffill(limit=max_forward_fill_days).dropna(how="any")
    if cleaned.empty:
        raise ValueError("No complete price observations remain after cleaning.")
    return cleaned


def save_price_snapshot(prices: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(path, index_label="date", float_format="%.10f")


def load_price_snapshot(path: str | Path) -> pd.DataFrame:
    prices = pd.read_csv(path, index_col="date", parse_dates=True)
    prices.index = pd.to_datetime(prices.index)
    return prices.sort_index()


def load_or_download_snapshot(
    path: str | Path,
    tickers: list[str],
    start: str,
    end: str,
    refresh: bool = False,
) -> pd.DataFrame:
    path = Path(path)
    if path.exists() and not refresh:
        prices = load_price_snapshot(path)
    else:
        prices = download_prices(tickers, start=start, end=end)
        save_price_snapshot(prices, path)

    unexpected = [column for column in prices.columns if column not in tickers]
    missing = [ticker for ticker in tickers if ticker not in prices.columns]
    if unexpected or missing:
        raise ValueError(f"Snapshot columns differ from protocol. Missing={missing}, unexpected={unexpected}")
    if prices.index.max() >= pd.Timestamp(end):
        raise ValueError("Snapshot contains an observation at or beyond the exclusive end date.")
    return clean_prices(prices[tickers])


def calculate_returns(prices: pd.DataFrame) -> pd.DataFrame:
    returns = prices.pct_change(fill_method=None).dropna(how="any")
    if returns.isna().any().any():
        raise ValueError("Returns contain missing values.")
    if not returns.index.is_monotonic_increasing:
        raise ValueError("Return dates must be increasing.")
    return returns


def data_quality_report(raw_prices: pd.DataFrame, cleaned_prices: pd.DataFrame) -> pd.DataFrame:
    rows = []
    aligned_cleaned = cleaned_prices.reindex(raw_prices.index)
    for ticker in raw_prices.columns:
        series = raw_prices[ticker]
        cleaned = aligned_cleaned[ticker]
        rows.append(
            {
                "ticker": ticker,
                "raw_start": series.first_valid_index(),
                "raw_end": series.last_valid_index(),
                "raw_missing_observations": int(series.isna().sum()),
                "clean_start": cleaned.first_valid_index(),
                "clean_end": cleaned.last_valid_index(),
                "clean_observations": int(cleaned.notna().sum()),
                "imputed_observations": int((series.isna() & cleaned.notna()).sum()),
                "non_positive_prices": int((series.dropna() <= 0).sum()),
            }
        )
    return pd.DataFrame(rows)
