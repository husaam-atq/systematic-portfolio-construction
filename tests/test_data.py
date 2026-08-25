from __future__ import annotations

import pandas as pd
import pytest

from src.data import calculate_returns, clean_prices, data_quality_report


def test_data_chronology_and_missing_values(synthetic_prices: pd.DataFrame) -> None:
    prices = synthetic_prices.copy()
    prices.iloc[20, 2] = float("nan")
    cleaned = clean_prices(prices)
    returns = calculate_returns(cleaned)
    assert returns.index.is_monotonic_increasing
    assert returns.index.is_unique
    assert not returns.isna().any().any()


def test_duplicate_dates_are_rejected(synthetic_prices: pd.DataFrame) -> None:
    duplicated = pd.concat([synthetic_prices.iloc[:5], synthetic_prices.iloc[[4]]])
    with pytest.raises(ValueError, match="unique"):
        clean_prices(duplicated)


def test_data_quality_reports_limited_imputation(synthetic_prices: pd.DataFrame) -> None:
    raw = synthetic_prices.iloc[:10].copy()
    raw.iloc[4, 0] = float("nan")
    cleaned = clean_prices(raw)
    report = data_quality_report(raw, cleaned).set_index("ticker")
    assert report.loc[raw.columns[0], "raw_missing_observations"] == 1
    assert report.loc[raw.columns[0], "imputed_observations"] == 1
