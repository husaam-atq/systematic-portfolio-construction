from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


ASSET_CLASS_MAP: dict[str, str] = {
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

STATIC_METHODS = [
    "Equal Weight",
    "Traditional 60/40",
    "Inverse Volatility",
    "Minimum Variance",
    "Minimum Variance Shrinkage",
    "Maximum Sharpe",
    "Maximum Sharpe Shrinkage",
    "Risk Parity",
    "Risk Parity Shrinkage",
    "Hierarchical Risk Parity",
]

TACTICAL_METHODS = [
    "Trend Filtered Equal Weight",
    "Trend Filtered Minimum Variance",
    "Trend Filtered Risk Parity",
    "Dual Momentum Equal Weight",
    "Dual Momentum Inverse Volatility",
]

VOL_TARGET_BASE_METHODS = [
    "Equal Weight",
    "Minimum Variance",
    "Maximum Sharpe",
    "Risk Parity",
]

CONFIRMATION_METHODS = [
    "Equal Weight",
    "Traditional 60/40",
    "Inverse Volatility",
    "Minimum Variance",
    "Minimum Variance Shrinkage",
    "Risk Parity",
    "Risk Parity Shrinkage",
    "Hierarchical Risk Parity",
    "Maximum Sharpe",
    "Maximum Sharpe Shrinkage",
    "Dual Momentum Equal Weight",
    "Maximum Sharpe Vol Target 10%",
]


@dataclass(frozen=True)
class ResearchConfig:
    development_start: str = "2010-01-01"
    development_end_exclusive: str = "2025-01-01"
    confirmation_start: str = "2025-01-01"
    confirmation_end_exclusive: str = "2026-08-25"
    estimation_window: int = 252
    rebalance_frequency: str = "monthly"
    max_asset_weight: float = 0.40
    covariance_shrinkage: float = 0.25
    transaction_cost_bps: float = 5.0
    cash_proxy: str = "SHY"
    financing_spread_annual: float = 0.005
    volatility_window: int = 63
    volatility_target: float = 0.10
    volatility_max_leverage: float = 1.50
    trend_moving_average_days: int = 200
    momentum_lookback_days: int = 252
    momentum_skip_days: int = 21
    momentum_top_n: int = 3
    bootstrap_block_days: int = 21
    bootstrap_resamples: int = 500
    bootstrap_seed: int = 20260825

    @property
    def tickers(self) -> list[str]:
        return list(ASSET_CLASS_MAP)

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["tickers"] = self.tickers
        result["static_methods"] = STATIC_METHODS
        result["tactical_methods"] = TACTICAL_METHODS
        result["vol_target_base_methods"] = VOL_TARGET_BASE_METHODS
        result["confirmation_methods"] = CONFIRMATION_METHODS
        return result


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "config"
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
REPORT_DIR = ROOT / "reports"
FIGURE_DIR = OUTPUT_DIR / "figures"
DEVELOPMENT_PRICE_FILE = DATA_DIR / "development_prices.csv"
CONFIRMATION_PRICE_FILE = DATA_DIR / "confirmation_prices.csv"
FROZEN_PROTOCOL_FILE = CONFIG_DIR / "frozen_protocol_v1.json"
FROZEN_HASH_FILE = CONFIG_DIR / "frozen_protocol_v1.sha256"
