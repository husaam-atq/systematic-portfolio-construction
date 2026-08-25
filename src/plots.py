from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.metrics import drawdown_series
from src.simulation import SimulationResult


plt.switch_backend("Agg")

REPRESENTATIVE_METHODS = [
    "Equal Weight",
    "Traditional 60/40",
    "Minimum Variance",
    "Risk Parity",
    "Hierarchical Risk Parity",
    "Maximum Sharpe",
    "Dual Momentum Equal Weight",
    "Maximum Sharpe Vol Target 10%",
]


def _save(path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def _selected_returns(simulations: dict[str, SimulationResult]) -> pd.DataFrame:
    methods = [method for method in REPRESENTATIVE_METHODS if method in simulations]
    return pd.DataFrame({method: simulations[method].performance["net_return"] for method in methods})


def plot_equity_curves(simulations: dict[str, SimulationResult], path: str | Path, title: str) -> None:
    returns = _selected_returns(simulations)
    equity = (1.0 + returns.fillna(0.0)).cumprod()
    plt.figure(figsize=(12, 6.5))
    for method in equity:
        plt.plot(equity.index, equity[method], label=method, linewidth=1.6)
    plt.title(title)
    plt.ylabel("Growth of $1")
    plt.grid(alpha=0.25)
    plt.legend(frameon=False, ncol=2, fontsize=8)
    _save(path)


def plot_drawdowns(simulations: dict[str, SimulationResult], path: str | Path, title: str) -> None:
    returns = _selected_returns(simulations)
    plt.figure(figsize=(12, 6.5))
    for method in returns:
        drawdowns = drawdown_series(returns[method])
        plt.plot(drawdowns.index, drawdowns, label=method, linewidth=1.4)
    plt.title(title)
    plt.ylabel("Drawdown")
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0%}"))
    plt.grid(alpha=0.25)
    plt.legend(frameon=False, ncol=2, fontsize=8)
    _save(path)


def plot_average_weights(simulations: dict[str, SimulationResult], path: str | Path) -> None:
    rows = []
    for method in REPRESENTATIVE_METHODS:
        if method not in simulations:
            continue
        active = simulations[method].post_trade_weights.loc[
            simulations[method].performance["active"] > 0.0
        ]
        for asset, value in active.mean().items():
            rows.append({"strategy": method, "asset": asset, "average_weight": value})
    table = pd.DataFrame(rows).pivot(index="strategy", columns="asset", values="average_weight").fillna(0.0)
    ax = table.plot(kind="bar", stacked=True, figsize=(12, 6.5), width=0.8)
    ax.set_title("Development Average Holdings Including Cash")
    ax.set_ylabel("Average portfolio weight")
    ax.set_xlabel("")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0%}"))
    ax.legend(frameon=False, bbox_to_anchor=(1.02, 1.0), loc="upper left", fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    _save(path)


def plot_risk_contributions(risk_summary: pd.DataFrame, path: str | Path) -> None:
    selected = risk_summary[risk_summary["strategy"].isin(REPRESENTATIVE_METHODS)]
    table = selected.pivot(
        index="strategy", columns="asset", values="average_percentage_risk_contribution"
    ).fillna(0.0)
    ax = table.plot(kind="bar", stacked=True, figsize=(12, 6.5), width=0.8)
    ax.set_title("Average Ex-Ante Percentage Risk Contribution")
    ax.set_ylabel("Contribution to portfolio variance")
    ax.set_xlabel("")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0%}"))
    ax.legend(frameon=False, bbox_to_anchor=(1.02, 1.0), loc="upper left", fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    _save(path)


def plot_effective_diversification(risk_summary: pd.DataFrame, path: str | Path) -> None:
    table = (
        risk_summary.groupby("strategy", as_index=True)
        .agg(
            effective_assets_by_weight=("effective_assets_by_weight", "first"),
            effective_assets_by_risk=("effective_assets_by_risk", "first"),
        )
        .reindex([method for method in REPRESENTATIVE_METHODS if method in risk_summary["strategy"].unique()])
    )
    ax = table.plot(kind="bar", figsize=(12, 6), color=["#4C78A8", "#F58518"])
    ax.set_title("Effective Diversification by Weight and Risk")
    ax.set_ylabel("Effective number of assets")
    ax.set_xlabel("")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    _save(path)


def plot_rolling_volatility(simulations: dict[str, SimulationResult], path: str | Path, window: int = 63) -> None:
    returns = _selected_returns(simulations)
    rolling = returns.rolling(window).std(ddof=0) * np.sqrt(252.0)
    plt.figure(figsize=(12, 6.5))
    for method in rolling:
        plt.plot(rolling.index, rolling[method], label=method, linewidth=1.3)
    plt.title(f"Rolling {window}-Day Realised Volatility")
    plt.ylabel("Annualised volatility")
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0%}"))
    plt.grid(alpha=0.25)
    plt.legend(frameon=False, ncol=2, fontsize=8)
    _save(path)


def plot_rolling_allocation(simulation: SimulationResult, path: str | Path, strategy: str) -> None:
    active = simulation.post_trade_weights.loc[simulation.performance["active"] > 0.0]
    monthly = active.resample("ME").last()
    asset_weights = monthly.drop(columns="CASH", errors="ignore")
    _, ax = plt.subplots(figsize=(12, 6.5))
    ax.stackplot(
        monthly.index,
        *[asset_weights[column].to_numpy() for column in asset_weights],
        labels=asset_weights.columns,
        linewidth=0.2,
    )
    if "CASH" in monthly:
        ax.plot(monthly.index, monthly["CASH"], color="black", linewidth=1.1, label="CASH (signed)")
    ax.axhline(0.0, color="black", linewidth=0.6)
    ax.set_title(f"Rolling Asset Allocation: {strategy}")
    ax.set_ylabel("Portfolio weight")
    ax.set_xlabel("")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0%}"))
    ax.legend(frameon=False, bbox_to_anchor=(1.02, 1.0), loc="upper left", fontsize=8)
    _save(path)


def plot_turnover_cost_sensitivity(sensitivity: pd.DataFrame, path: str | Path) -> None:
    selected = sensitivity[
        (sensitivity["dimension"] == "transaction_cost_bps")
        & sensitivity["strategy"].isin(["Equal Weight", "Minimum Variance", "Maximum Sharpe", "Risk Parity", "Hierarchical Risk Parity"])
    ]
    pivot = selected.pivot(index="value", columns="strategy", values="cagr")
    plt.figure(figsize=(11, 6))
    for method in pivot:
        plt.plot(pivot.index.astype(float), pivot[method], marker="o", label=method)
    plt.title("Transaction-Cost Sensitivity")
    plt.xlabel("One-way transaction cost (bps)")
    plt.ylabel("CAGR")
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.1%}"))
    plt.grid(alpha=0.25)
    plt.legend(frameon=False, fontsize=8)
    _save(path)


def plot_estimation_window_sensitivity(sensitivity: pd.DataFrame, path: str | Path) -> None:
    selected = sensitivity[sensitivity["dimension"] == "estimation_window"]
    pivot = selected.pivot(index="value", columns="strategy", values="sharpe_ratio")
    plt.figure(figsize=(11, 6))
    for method in pivot:
        plt.plot(pivot.index.astype(float), pivot[method], marker="o", label=method)
    plt.title("Estimation-Window Sensitivity")
    plt.xlabel("Trailing estimation window (trading days)")
    plt.ylabel("Development Sharpe ratio")
    plt.grid(alpha=0.25)
    plt.legend(frameon=False, fontsize=8)
    _save(path)


def plot_stress_comparison(stress: pd.DataFrame, path: str | Path) -> None:
    selected = stress[stress["strategy"].isin(REPRESENTATIVE_METHODS)]
    table = selected.pivot(index="strategy", columns="episode", values="total_return")
    ax = table.plot(kind="bar", figsize=(14, 7), width=0.82)
    ax.set_title("Predefined Stress-Episode Returns")
    ax.set_ylabel("Episode return")
    ax.set_xlabel("")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0%}"))
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, bbox_to_anchor=(1.02, 1.0), loc="upper left", fontsize=8)
    _save(path)


def plot_benchmark_relative(relative: pd.DataFrame, path: str | Path) -> None:
    selected = relative[
        (relative["benchmark"] == "Equal Weight") & relative["strategy"].isin(REPRESENTATIVE_METHODS)
    ].set_index("strategy")
    ax = selected["annualized_excess_return"].plot(kind="bar", figsize=(11, 6), color="#4C78A8")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title("Annualised Excess Return versus Equal Weight")
    ax.set_ylabel("Annualised excess return")
    ax.set_xlabel("")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.1%}"))
    ax.grid(axis="y", alpha=0.25)
    _save(path)


def generate_development_figures(
    simulations: dict[str, SimulationResult],
    risk_summary: pd.DataFrame,
    sensitivity: pd.DataFrame,
    stress: pd.DataFrame,
    relative: pd.DataFrame,
    output_dir: str | Path,
) -> None:
    output_dir = Path(output_dir)
    plot_equity_curves(simulations, output_dir / "development_equity.png", "Development Equity Curves (2010-2024)")
    plot_drawdowns(simulations, output_dir / "development_drawdown.png", "Development Drawdowns (2010-2024)")
    plot_average_weights(simulations, output_dir / "average_weights.png")
    plot_risk_contributions(risk_summary, output_dir / "risk_contributions.png")
    plot_effective_diversification(risk_summary, output_dir / "effective_diversification.png")
    plot_rolling_volatility(simulations, output_dir / "rolling_portfolio_volatility.png")
    allocation_method = "Risk Parity" if "Risk Parity" in simulations else next(iter(simulations))
    plot_rolling_allocation(simulations[allocation_method], output_dir / "rolling_asset_allocation.png", allocation_method)
    plot_turnover_cost_sensitivity(sensitivity, output_dir / "turnover_cost_sensitivity.png")
    plot_estimation_window_sensitivity(sensitivity, output_dir / "estimation_window_sensitivity.png")
    plot_stress_comparison(stress, output_dir / "stress_period_comparison.png")
    plot_benchmark_relative(relative, output_dir / "benchmark_relative_performance.png")


def generate_confirmation_figures(
    simulations: dict[str, SimulationResult],
    output_dir: str | Path,
) -> None:
    output_dir = Path(output_dir)
    plot_equity_curves(simulations, output_dir / "confirmation_equity.png", "Frozen Confirmation Equity Curves")
    plot_drawdowns(simulations, output_dir / "confirmation_drawdown.png", "Frozen Confirmation Drawdowns")
