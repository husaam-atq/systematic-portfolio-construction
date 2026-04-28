from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.metrics import drawdown_series


plt.switch_backend("Agg")


def _save(path: str | Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def plot_equity_curves(daily_returns: pd.DataFrame, path: str | Path) -> None:
    equity = (1.0 + daily_returns.fillna(0.0)).cumprod()
    plt.figure(figsize=(11, 6))
    for column in equity.columns:
        plt.plot(equity.index, equity[column], label=column, linewidth=1.7)
    plt.title("Equity Curves")
    plt.ylabel("Growth of $1")
    plt.xlabel("")
    plt.grid(alpha=0.25)
    plt.legend(frameon=False, ncol=2)
    _save(path)


def plot_drawdowns(daily_returns: pd.DataFrame, path: str | Path) -> None:
    plt.figure(figsize=(11, 6))
    for column in daily_returns.columns:
        drawdown = drawdown_series(daily_returns[column])
        plt.plot(drawdown.index, drawdown, label=column, linewidth=1.5)
    plt.title("Drawdowns")
    plt.ylabel("Drawdown")
    plt.xlabel("")
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0%}"))
    plt.grid(alpha=0.25)
    plt.legend(frameon=False, ncol=2)
    _save(path)


def plot_rolling_sharpe(daily_returns: pd.DataFrame, path: str | Path, window: int = 126) -> None:
    rolling = daily_returns.rolling(window).mean() / daily_returns.rolling(window).std(ddof=0)
    rolling = rolling * np.sqrt(252)
    plt.figure(figsize=(11, 6))
    for column in rolling.columns:
        plt.plot(rolling.index, rolling[column], label=column, linewidth=1.4)
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.title("Rolling 126-Day Sharpe Ratio")
    plt.ylabel("Sharpe")
    plt.xlabel("")
    plt.grid(alpha=0.25)
    plt.legend(frameon=False, ncol=2)
    _save(path)


def plot_average_weights(average_weights: pd.DataFrame, path: str | Path) -> None:
    table = average_weights.pivot(index="strategy", columns="asset", values="average_weight").fillna(0.0)
    ax = table.plot(kind="bar", stacked=True, figsize=(11, 6), width=0.78)
    ax.set_title("Average Portfolio Weights")
    ax.set_ylabel("Average weight")
    ax.set_xlabel("")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0%}"))
    ax.legend(frameon=False, bbox_to_anchor=(1.02, 1.0), loc="upper left")
    ax.grid(axis="y", alpha=0.25)
    _save(path)


def plot_risk_contribution(risk_contribution: pd.DataFrame, path: str | Path) -> None:
    table = risk_contribution.pivot(
        index="strategy",
        columns="asset",
        values="average_component_risk_contribution",
    ).fillna(0.0)
    ax = table.plot(kind="bar", stacked=True, figsize=(11, 6), width=0.78)
    ax.set_title("Average Component Risk Contribution")
    ax.set_ylabel("Share of portfolio variance")
    ax.set_xlabel("")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0%}"))
    ax.legend(frameon=False, bbox_to_anchor=(1.02, 1.0), loc="upper left")
    ax.grid(axis="y", alpha=0.25)
    _save(path)


def plot_cost_sensitivity(cost_sensitivity: pd.DataFrame, path: str | Path) -> None:
    pivot = cost_sensitivity.pivot(index="transaction_cost_bps", columns="strategy", values="cagr")
    plt.figure(figsize=(11, 6))
    for column in pivot.columns:
        plt.plot(pivot.index, pivot[column], marker="o", label=column, linewidth=1.7)
    plt.title("Transaction Cost Sensitivity")
    plt.ylabel("CAGR")
    plt.xlabel("Transaction cost, bps per one-way trade")
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.1%}"))
    plt.grid(alpha=0.25)
    plt.legend(frameon=False, ncol=2)
    _save(path)


def plot_performance_comparison(performance: pd.DataFrame, path: str | Path) -> None:
    fields = ["cagr", "sharpe_ratio", "max_drawdown"]
    labels = ["CAGR", "Sharpe", "Max drawdown"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, field, label in zip(axes, fields, labels, strict=True):
        values = performance[field].copy()
        values.plot(kind="bar", ax=ax, color="#4C78A8", width=0.78)
        ax.set_title(label)
        ax.set_xlabel("")
        ax.grid(axis="y", alpha=0.25)
        if field != "sharpe_ratio":
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0%}"))
        ax.tick_params(axis="x", rotation=35)
    fig.suptitle("Performance Comparison", y=1.02)
    _save(path)
