from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path

import pandas as pd

from src.backtest import (
    add_volatility_target_simulations,
    daily_return_frame,
    daily_weight_frame,
    generate_target_weights,
    simulate_strategies,
)
from src.config import (
    CONFIRMATION_PRICE_FILE,
    DEVELOPMENT_PRICE_FILE,
    FIGURE_DIR,
    FROZEN_HASH_FILE,
    FROZEN_PROTOCOL_FILE,
    OUTPUT_DIR,
    REPORT_DIR,
    ROOT,
    ResearchConfig,
)
from src.data import (
    calculate_returns,
    clean_prices,
    data_quality_report,
    load_or_download_snapshot,
    load_price_snapshot,
)
from src.freeze import load_and_verify_protocol, sha256_file
from src.metrics import benchmark_relative_metrics, capture_ratios, performance_summary
from src.plots import generate_confirmation_figures
from src.reporting import write_confirmation_report, write_readme
from src.simulation import slice_simulation


def _save(frame: pd.DataFrame, name: str, index: bool = False) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUT_DIR / name, index=index)


def _config_from_protocol(protocol: dict[str, object]) -> ResearchConfig:
    allowed = {field.name for field in fields(ResearchConfig)}
    values = protocol.get("research_config")
    if not isinstance(values, dict) or set(values) != allowed:
        raise RuntimeError("Frozen research_config fields do not match ResearchConfig.")
    return ResearchConfig(**values)


def run_confirmation(refresh_data: bool = False) -> None:
    protocol, freeze_hash = load_and_verify_protocol(
        FROZEN_PROTOCOL_FILE,
        FROZEN_HASH_FILE,
        ROOT,
    )
    config = _config_from_protocol(protocol)
    methods = protocol.get("confirmation_methods")
    if not isinstance(methods, list) or not methods:
        raise RuntimeError("Frozen confirmation method list is missing.")

    raw_development = load_price_snapshot(DEVELOPMENT_PRICE_FILE)
    if raw_development.index.max() >= pd.Timestamp(config.confirmation_start):
        raise RuntimeError("Development snapshot crosses the frozen confirmation boundary.")
    if sha256_file(DEVELOPMENT_PRICE_FILE) != protocol.get("development_price_sha256"):
        raise RuntimeError("Development price snapshot differs from the frozen protocol.")

    confirmation_prices = load_or_download_snapshot(
        CONFIRMATION_PRICE_FILE,
        config.tickers,
        start=config.confirmation_start,
        end=config.confirmation_end_exclusive,
        refresh=refresh_data,
    )
    raw_confirmation = load_price_snapshot(CONFIRMATION_PRICE_FILE)
    if raw_confirmation.index.min() < pd.Timestamp(config.confirmation_start):
        raise RuntimeError("Confirmation snapshot contains pre-confirmation observations.")

    development_prices = clean_prices(raw_development[config.tickers])
    prices = pd.concat([development_prices, confirmation_prices]).sort_index()
    prices = prices.loc[~prices.index.duplicated(keep="last")]
    prices = clean_prices(prices[config.tickers])
    asset_returns = calculate_returns(prices)

    overlay_names = {method for method in methods if "Vol Target" in method}
    base_methods = [method for method in methods if method not in overlay_names]
    targets, _, _ = generate_target_weights(
        asset_returns,
        prices,
        config,
        methods=base_methods,
    )
    simulations = simulate_strategies(asset_returns, targets, config)
    simulations, _ = add_volatility_target_simulations(simulations, asset_returns, config)
    missing = [method for method in methods if method not in simulations]
    if missing:
        raise RuntimeError(f"Frozen confirmation methods were not produced: {missing}")

    confirmation = {
        method: slice_simulation(
            simulations[method],
            config.confirmation_start,
            pd.Timestamp(config.confirmation_end_exclusive) - pd.Timedelta(days=1),
        )
        for method in methods
    }
    returns = daily_return_frame(confirmation)
    if returns.empty or returns.index.min() < pd.Timestamp(config.confirmation_start):
        raise RuntimeError("Confirmation return interval is invalid.")
    cash_returns = asset_returns.loc[returns.index, config.cash_proxy]
    performance = performance_summary(confirmation, cash_returns=cash_returns)
    relative = benchmark_relative_metrics(returns, asset_returns.loc[returns.index, "SPY"])
    relative = relative.merge(
        capture_ratios(returns, asset_returns.loc[returns.index, "SPY"]),
        on="strategy",
        how="left",
    )

    _save(performance.reset_index(), "confirmation_performance.csv")
    _save(returns.reset_index(names="date"), "confirmation_daily_returns.csv")
    _save(
        daily_weight_frame(confirmation).reset_index(names="date"),
        "confirmation_portfolio_weights.csv",
    )
    _save(relative, "confirmation_benchmark_relative_metrics.csv")
    _save(
        data_quality_report(raw_confirmation, confirmation_prices),
        "confirmation_data_quality.csv",
    )

    run_record = {
        "frozen_protocol_sha256": freeze_hash,
        "confirmation_price_sha256": sha256_file(CONFIRMATION_PRICE_FILE),
        "first_observation": returns.index.min().strftime("%Y-%m-%d"),
        "last_observation": returns.index.max().strftime("%Y-%m-%d"),
        "observations": int(len(returns)),
        "methods": methods,
    }
    (OUTPUT_DIR / "confirmation_run.json").write_text(
        json.dumps(run_record, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    generate_confirmation_figures(confirmation, FIGURE_DIR)
    write_confirmation_report(
        REPORT_DIR / "confirmation_results.md",
        performance,
        freeze_hash,
        returns.index.min(),
        returns.index.max(),
    )
    development_performance = pd.read_csv(
        OUTPUT_DIR / "development_performance.csv",
        index_col="strategy",
    )
    development_protocol = json.loads(
        (OUTPUT_DIR / "development_protocol.json").read_text(encoding="utf-8")
    )
    write_readme(
        Path("README.md"),
        development_performance,
        performance,
        freeze_hash,
        development_evaluation_start=development_protocol["common_evaluation_start"],
        confirmation_end=returns.index.max(),
        baseline_performance=pd.read_csv(
            ROOT / "archive/baseline_48bcbe9/tables/performance_summary.csv",
            index_col="strategy",
        ),
        development_risk_summary=pd.read_csv(OUTPUT_DIR / "risk_contribution_summary.csv"),
        development_uncertainty=pd.read_csv(OUTPUT_DIR / "bootstrap_uncertainty.csv"),
        development_stability=pd.read_csv(OUTPUT_DIR / "optimizer_stability.csv"),
    )

    print(f"Frozen confirmation completed through {returns.index.max().date()}.")
    print(
        performance[
            ["cagr", "sharpe_ratio", "annualized_volatility", "max_drawdown", "average_monthly_turnover"]
        ].to_string(float_format=lambda value: f"{value:0.4f}")
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the hash-verified frozen 2025+ confirmation.")
    parser.add_argument("--refresh-data", action="store_true")
    args = parser.parse_args()
    run_confirmation(refresh_data=args.refresh_data)


if __name__ == "__main__":
    main()
