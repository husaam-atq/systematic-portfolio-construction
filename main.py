from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.allocation import covariance_matrix
from src.backtest import (
    add_volatility_target_simulations,
    daily_return_frame,
    daily_weight_frame,
    generate_target_weights,
    rebalance_dates,
    simulate_strategies,
)
from src.config import (
    DEVELOPMENT_PRICE_FILE,
    FIGURE_DIR,
    OUTPUT_DIR,
    REPORT_DIR,
    ResearchConfig,
)
from src.data import (
    calculate_returns,
    data_quality_report,
    load_or_download_snapshot,
    load_price_snapshot,
)
from src.metrics import (
    benchmark_relative_metrics,
    capture_ratios,
    moving_block_bootstrap,
    performance_summary,
    segment_results,
    stress_results,
)
from src.plots import generate_development_figures
from src.reporting import (
    write_confirmation_report,
    write_development_report,
    write_experiment_log,
    write_limitations,
    write_readme,
)
from src.risk import (
    drawdown_contribution_summary,
    return_contribution_summary,
    risk_parity_verification,
    risk_snapshot,
    summarize_risk_records,
)
from src.simulation import common_start_targets
from src.robustness import (
    input_perturbation_sensitivity,
    optimizer_stability_summary,
    parameter_sensitivity,
)


def _save(frame: pd.DataFrame, name: str, index: bool = False) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUT_DIR / name, index=index)


def _flatten_targets(targets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    combined = pd.concat(targets, axis=1)
    combined.columns = [f"{strategy}__{asset}" for strategy, asset in combined.columns]
    return combined


def _volatility_target_risk_records(
    simulations: dict,
    asset_returns: pd.DataFrame,
    config: ResearchConfig,
) -> pd.DataFrame:
    records = []
    dates = rebalance_dates(asset_returns.index, "monthly")
    for strategy, simulation in simulations.items():
        if "Vol Target" not in strategy:
            continue
        for date in dates:
            position = asset_returns.index.get_loc(date)
            if position + 1 < config.estimation_window:
                continue
            weights = simulation.post_trade_weights.loc[date, asset_returns.columns]
            if weights.abs().sum() <= 0.0:
                continue
            trailing = asset_returns.iloc[position - config.estimation_window + 1 : position + 1]
            snapshot = risk_snapshot(weights, covariance_matrix(trailing))
            snapshot.insert(0, "strategy", strategy)
            snapshot.insert(0, "date", date)
            records.extend(snapshot.to_dict(orient="records"))
    return pd.DataFrame(records)


def run_development(refresh_data: bool = False) -> None:
    config = ResearchConfig()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    prices = load_or_download_snapshot(
        DEVELOPMENT_PRICE_FILE,
        config.tickers,
        start=config.development_start,
        end=config.development_end_exclusive,
        refresh=refresh_data,
    )
    if prices.index.max() >= pd.Timestamp(config.confirmation_start):
        raise RuntimeError("Development data crossed the sealed confirmation boundary.")
    asset_returns = calculate_returns(prices)

    targets, diagnostics, risk_records = generate_target_weights(asset_returns, prices, config)
    base_simulations = simulate_strategies(asset_returns, targets, config)
    simulations, vol_metadata = add_volatility_target_simulations(
        dict(base_simulations), asset_returns, config
    )
    evaluation_start = max(
        simulation.performance["net_return"].first_valid_index()
        for simulation in simulations.values()
    )
    analysis_targets = common_start_targets(simulations, asset_returns.columns, evaluation_start)
    analysis_simulations = simulate_strategies(
        asset_returns.loc[evaluation_start:],
        analysis_targets,
        config,
    )
    vol_risk = _volatility_target_risk_records(simulations, asset_returns, config)
    if not vol_risk.empty:
        risk_records = pd.concat([risk_records, vol_risk], ignore_index=True)

    risk_records = risk_records[pd.to_datetime(risk_records["date"]) >= evaluation_start].copy()
    returns = daily_return_frame(analysis_simulations)
    cash_returns = asset_returns.loc[evaluation_start:, config.cash_proxy]
    performance = performance_summary(analysis_simulations, cash_returns=cash_returns)
    risk_summary = summarize_risk_records(risk_records)
    relative = benchmark_relative_metrics(returns, asset_returns["SPY"])
    captures = capture_ratios(returns, asset_returns["SPY"])
    relative = relative.merge(captures, on="strategy", how="left")
    stress = stress_results(returns)
    segments = segment_results(analysis_simulations, risk_records, cash_returns=cash_returns)
    parity_check = risk_parity_verification(risk_records)
    perturbations = input_perturbation_sensitivity(
        asset_returns,
        rebalance_dates(asset_returns.index, config.rebalance_frequency),
        config,
    )
    stability = optimizer_stability_summary(diagnostics, analysis_simulations, perturbations)
    sensitivity = parameter_sensitivity(asset_returns, prices, config, targets, base_simulations)
    uncertainty = moving_block_bootstrap(
        returns,
        block_days=config.bootstrap_block_days,
        resamples=config.bootstrap_resamples,
        seed=config.bootstrap_seed,
        cash_returns=cash_returns,
    )
    return_contributions = return_contribution_summary(analysis_simulations)
    drawdown_contributions = drawdown_contribution_summary(analysis_simulations)
    quality = data_quality_report(load_price_snapshot(DEVELOPMENT_PRICE_FILE), prices)

    _save(performance.reset_index(), "development_performance.csv")
    _save(returns.reset_index(names="date"), "development_daily_returns.csv")
    _save(daily_weight_frame(analysis_simulations).reset_index(names="date"), "development_portfolio_weights.csv")
    _save(_flatten_targets(targets).reset_index(names="date"), "development_target_weights.csv")
    _save(diagnostics, "optimizer_diagnostics.csv")
    _save(stability, "optimizer_stability.csv")
    _save(perturbations, "input_perturbation_sensitivity.csv")
    _save(risk_records, "risk_contribution_history.csv")
    _save(risk_summary, "risk_contribution_summary.csv")
    _save(parity_check, "risk_parity_verification.csv")
    _save(return_contributions, "return_contributions.csv")
    _save(drawdown_contributions, "drawdown_contributions.csv")
    _save(relative, "benchmark_relative_metrics.csv")
    _save(stress, "stress_results.csv")
    _save(segments, "walk_forward_segments.csv")
    _save(sensitivity, "parameter_sensitivity.csv")
    _save(uncertainty, "bootstrap_uncertainty.csv")
    _save(vol_metadata, "vol_target_metadata.csv")
    _save(quality, "data_quality.csv")
    protocol = config.to_dict()
    protocol["common_evaluation_start"] = evaluation_start.strftime("%Y-%m-%d")
    (OUTPUT_DIR / "development_protocol.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True), encoding="utf-8"
    )

    generate_development_figures(
        analysis_simulations,
        risk_summary,
        sensitivity,
        stress,
        relative,
        FIGURE_DIR,
    )
    write_development_report(
        REPORT_DIR / "development_results.md",
        performance,
        segments,
        uncertainty,
        stability,
        parity_check,
        sensitivity,
        evaluation_start,
    )
    write_experiment_log(REPORT_DIR / "experiment_log.md", sensitivity, performance)
    write_limitations(REPORT_DIR / "limitations.md")
    write_confirmation_report(REPORT_DIR / "confirmation_results.md")
    write_readme(
        Path("README.md"),
        performance,
        development_evaluation_start=evaluation_start,
        baseline_performance=pd.read_csv(
            Path("archive/baseline_48bcbe9/tables/performance_summary.csv"),
            index_col="strategy",
        ),
        development_risk_summary=risk_summary,
        development_uncertainty=uncertainty,
        development_stability=stability,
    )

    print("Development research completed without requesting 2025+ data.")
    print(performance[["cagr", "sharpe_ratio", "annualized_volatility", "max_drawdown", "average_monthly_turnover"]].to_string(float_format=lambda value: f"{value:0.4f}"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the sealed 2010-2024 development research pipeline.")
    parser.add_argument(
        "--refresh-data",
        action="store_true",
        help="Refresh only the development snapshot, with an exclusive 2025-01-01 end boundary.",
    )
    args = parser.parse_args()
    run_development(refresh_data=args.refresh_data)


if __name__ == "__main__":
    main()
