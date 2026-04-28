from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.backtest import generate_weight_decisions, run_all_backtests
from src.data import ETF_UNIVERSE, calculate_returns, download_prices
from src.metrics import performance_summary, subperiod_summary
from src.plots import (
    plot_average_weights,
    plot_cost_sensitivity,
    plot_drawdowns,
    plot_equity_curves,
    plot_performance_comparison,
    plot_risk_contribution,
    plot_rolling_sharpe,
)
from src.risk import summarize_average_weights, summarize_risk_contributions


START_DATE = "2010-01-01"
END_DATE = "2025-01-01"
ESTIMATION_WINDOW = 252
DEFAULT_TC_BPS = 5.0
COST_LEVELS_BPS = [0.0, 5.0, 10.0, 20.0]
MAX_WEIGHT = 0.40
OUTPUT_DIR = Path("outputs")


SUBPERIODS = {
    "2010-2014": ("2010-01-01", "2014-12-31"),
    "2015-2019": ("2015-01-01", "2019-12-31"),
    "2020-2021": ("2020-01-01", "2021-12-31"),
    "2022": ("2022-01-01", "2022-12-31"),
    "2023-2024": ("2023-01-01", "2024-12-31"),
    "Full sample": ("2010-01-01", "2024-12-31"),
}


def _flatten_weight_columns(weights: dict[str, pd.DataFrame]) -> pd.DataFrame:
    combined = pd.concat(weights, axis=1)
    combined.columns = [f"{strategy}__{asset}" for strategy, asset in combined.columns]
    return combined


def _format_percent(value: float, decimals: int = 2) -> str:
    if pd.isna(value):
        return ""
    return f"{value:.{decimals}%}"


def _format_number(value: float, decimals: int = 2) -> str:
    if pd.isna(value):
        return ""
    return f"{value:.{decimals}f}"


def _markdown_table(frame: pd.DataFrame, formats: dict[str, str]) -> str:
    formatted = frame.copy()
    for column, formatter in formats.items():
        if column not in formatted.columns:
            continue
        if formatter == "percent":
            formatted[column] = formatted[column].map(_format_percent)
        elif formatter == "number":
            formatted[column] = formatted[column].map(_format_number)
        elif formatter == "integer":
            formatted[column] = formatted[column].map(lambda value: f"{int(value)}" if pd.notna(value) else "")

    columns = [str(column) for column in formatted.columns]
    rows = [[str(value) for value in row] for row in formatted.to_numpy()]
    widths = [
        max(len(columns[index]), *(len(row[index]) for row in rows))
        for index in range(len(columns))
    ]

    header = "| " + " | ".join(column.ljust(widths[index]) for index, column in enumerate(columns)) + " |"
    separator = "| " + " | ".join("-" * widths[index] for index in range(len(columns))) + " |"
    body = [
        "| " + " | ".join(row[index].ljust(widths[index]) for index in range(len(columns))) + " |"
        for row in rows
    ]
    return "\n".join([header, separator, *body])


def _build_generated_readme_section(output_dir: Path) -> str:
    performance = pd.read_csv(output_dir / "performance_summary.csv", index_col=0)
    subperiods = pd.read_csv(output_dir / "subperiod_results.csv")
    costs = pd.read_csv(output_dir / "cost_sensitivity.csv")
    risk = pd.read_csv(output_dir / "risk_contribution.csv")

    perf_table = performance.reset_index()[[
        "strategy",
        "total_return",
        "cagr",
        "annualized_volatility",
        "sharpe_ratio",
        "max_drawdown",
        "average_turnover",
        "effective_number_of_assets",
    ]]

    subperiod_table = subperiods[[
        "subperiod",
        "strategy",
        "total_return",
        "cagr",
        "sharpe_ratio",
        "max_drawdown",
        "annualized_volatility",
    ]]

    cost_table = costs[[
        "transaction_cost_bps",
        "strategy",
        "cagr",
        "sharpe_ratio",
        "max_drawdown",
        "average_turnover",
    ]]

    risk_summary = (
        risk.groupby("strategy", as_index=False)
        .agg(
            average_portfolio_volatility=("average_portfolio_volatility", "mean"),
            risk_concentration=("risk_concentration", "first"),
            effective_number_of_risk_contributors=("effective_number_of_risk_contributors", "first"),
        )
    )

    best_sharpe = performance["sharpe_ratio"].idxmax()
    best_sharpe_value = performance.loc[best_sharpe, "sharpe_ratio"]
    equal_sharpe = performance.loc["Equal Weight", "sharpe_ratio"]
    equal_cagr = performance.loc["Equal Weight", "cagr"]
    best_cagr = performance["cagr"].idxmax()
    risk_parity_dd = performance.loc["Risk Parity", "max_drawdown"]
    equal_dd = performance.loc["Equal Weight", "max_drawdown"]
    inverse_vol = performance.loc["Inverse Volatility", "annualized_volatility"]
    equal_vol = performance.loc["Equal Weight", "annualized_volatility"]
    max_sharpe_turnover = performance.loc["Maximum Sharpe", "average_turnover"]
    equal_turnover = performance.loc["Equal Weight", "average_turnover"]

    cost_0 = costs[costs["transaction_cost_bps"] == 0.0].set_index("strategy")
    cost_20 = costs[costs["transaction_cost_bps"] == 20.0].set_index("strategy")
    cost_impact = (cost_20["cagr"] - cost_0["cagr"]).sort_values().iloc[0]

    if best_sharpe == "Equal Weight":
        optimization_text = "Equal weight had the highest Sharpe ratio in this run, so the optimized methods did not improve the main risk-adjusted result."
    elif best_sharpe_value > equal_sharpe:
        optimization_text = f"{best_sharpe} had the highest Sharpe ratio, improving on equal weight in risk-adjusted terms."
    else:
        optimization_text = "The optimized methods did not produce a clear Sharpe improvement over equal weight."

    if risk_parity_dd > equal_dd:
        drawdown_text = "Risk parity produced a shallower max drawdown than equal weight."
    else:
        drawdown_text = "Risk parity did not reduce max drawdown versus equal weight in this run."

    if inverse_vol < equal_vol:
        inverse_vol_text = "Inverse volatility reduced annualized volatility relative to equal weight."
    else:
        inverse_vol_text = "Inverse volatility did not reduce annualized volatility relative to equal weight."

    instability_text = (
        "Maximum Sharpe is the most estimation-sensitive method because it uses trailing expected returns as well as covariance. "
        f"Its average turnover was {_format_percent(max_sharpe_turnover)}, compared with {_format_percent(equal_turnover)} for equal weight."
    )

    generated = [
        "The tables below are regenerated by `python main.py` from the CSV files in `outputs/`.",
        "",
        "### Performance Summary",
        "",
        _markdown_table(
            perf_table,
            {
                "total_return": "percent",
                "cagr": "percent",
                "annualized_volatility": "percent",
                "sharpe_ratio": "number",
                "max_drawdown": "percent",
                "average_turnover": "percent",
                "effective_number_of_assets": "number",
            },
        ),
        "",
        "### Subperiod Results",
        "",
        _markdown_table(
            subperiod_table,
            {
                "total_return": "percent",
                "cagr": "percent",
                "sharpe_ratio": "number",
                "max_drawdown": "percent",
                "annualized_volatility": "percent",
            },
        ),
        "",
        "### Transaction Cost Sensitivity",
        "",
        _markdown_table(
            cost_table,
            {
                "cagr": "percent",
                "sharpe_ratio": "number",
                "max_drawdown": "percent",
                "average_turnover": "percent",
            },
        ),
        "",
        "### Risk Concentration Summary",
        "",
        _markdown_table(
            risk_summary,
            {
                "average_portfolio_volatility": "percent",
                "risk_concentration": "number",
                "effective_number_of_risk_contributors": "number",
            },
        ),
        "",
        "### Interpretation",
        "",
        f"- Best risk-adjusted return: {best_sharpe} had the highest Sharpe ratio ({best_sharpe_value:.2f}).",
        f"- Highest CAGR: {best_cagr} generated the highest CAGR ({_format_percent(performance.loc[best_cagr, 'cagr'])}); equal weight generated {_format_percent(equal_cagr)}.",
        f"- Optimization versus equal weight: {optimization_text}",
        f"- Risk parity and drawdowns: {drawdown_text}",
        f"- Inverse volatility: {inverse_vol_text}",
        f"- Transaction costs: moving from 0 bps to 20 bps reduced the most affected strategy CAGR by {_format_percent(abs(cost_impact))}.",
        f"- Optimization stability: {instability_text}",
    ]
    return "\n".join(generated)


def update_readme_results(output_dir: Path) -> None:
    readme_path = Path("README.md")
    start_marker = "<!-- RESULTS_START -->"
    end_marker = "<!-- RESULTS_END -->"
    readme = readme_path.read_text(encoding="utf-8")
    generated = _build_generated_readme_section(output_dir)

    before, _, remainder = readme.partition(start_marker)
    _, _, after = remainder.partition(end_marker)
    if not before or not after:
        raise ValueError("README result markers are missing.")

    readme_path.write_text(
        f"{before}{start_marker}\n{generated}\n{end_marker}{after}",
        encoding="utf-8",
    )


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    (OUTPUT_DIR / ".gitkeep").touch()

    tickers = list(ETF_UNIVERSE)
    prices = download_prices(tickers=tickers, start=START_DATE, end=END_DATE)
    returns = calculate_returns(prices).loc[: "2024-12-31"].dropna(how="any")

    decisions, risk_records = generate_weight_decisions(
        returns,
        estimation_window=ESTIMATION_WINDOW,
        max_weight=MAX_WEIGHT,
    )
    backtests, weights, daily_returns = run_all_backtests(returns, decisions, tc_bps=DEFAULT_TC_BPS)

    performance = performance_summary(backtests, weights)
    subperiods = subperiod_summary(daily_returns, SUBPERIODS)
    average_weights = summarize_average_weights(weights)
    risk_contribution = summarize_risk_contributions(risk_records)

    cost_rows = []
    for cost_bps in COST_LEVELS_BPS:
        cost_backtests, cost_weights, _ = run_all_backtests(returns, decisions, tc_bps=cost_bps)
        cost_summary = performance_summary(cost_backtests, cost_weights).reset_index()
        cost_summary.insert(0, "transaction_cost_bps", cost_bps)
        cost_rows.append(
            cost_summary[[
                "transaction_cost_bps",
                "strategy",
                "cagr",
                "sharpe_ratio",
                "max_drawdown",
                "average_turnover",
            ]]
        )
    cost_sensitivity = pd.concat(cost_rows, ignore_index=True)

    performance.to_csv(OUTPUT_DIR / "performance_summary.csv")
    subperiods.to_csv(OUTPUT_DIR / "subperiod_results.csv", index=False)
    cost_sensitivity.to_csv(OUTPUT_DIR / "cost_sensitivity.csv", index=False)
    risk_contribution.to_csv(OUTPUT_DIR / "risk_contribution.csv", index=False)
    average_weights.to_csv(OUTPUT_DIR / "average_weights.csv", index=False)
    daily_returns.to_csv(OUTPUT_DIR / "daily_returns.csv")
    _flatten_weight_columns(weights).to_csv(OUTPUT_DIR / "portfolio_weights.csv")

    plot_equity_curves(daily_returns, OUTPUT_DIR / "equity_curves.png")
    plot_drawdowns(daily_returns, OUTPUT_DIR / "drawdowns.png")
    plot_rolling_sharpe(daily_returns, OUTPUT_DIR / "rolling_sharpe.png")
    plot_average_weights(average_weights, OUTPUT_DIR / "average_weights.png")
    plot_risk_contribution(risk_contribution, OUTPUT_DIR / "risk_contribution.png")
    plot_cost_sensitivity(cost_sensitivity, OUTPUT_DIR / "cost_sensitivity.png")
    plot_performance_comparison(performance, OUTPUT_DIR / "performance_comparison.png")

    update_readme_results(OUTPUT_DIR)

    print("Generated allocation engine outputs in outputs/.")
    print(performance.to_string(float_format=lambda value: f"{value:0.4f}"))


if __name__ == "__main__":
    main()
