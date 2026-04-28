from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.backtest import apply_volatility_target, generate_weight_decisions, run_all_backtests
from src.data import ETF_UNIVERSE, calculate_returns, download_prices
from src.metrics import benchmark_relative_metrics, performance_summary, stress_results, subperiod_summary
from src.plots import (
    plot_average_weights,
    plot_cost_sensitivity,
    plot_drawdowns,
    plot_equity_curves,
    plot_performance_comparison,
    plot_risk_contribution,
    plot_rolling_sharpe,
    plot_strategy_group,
    plot_stress_period_comparison,
)
from src.risk import summarize_average_weights, summarize_risk_contributions


START_DATE = "2010-01-01"
END_DATE = "2025-01-01"
ESTIMATION_WINDOW = 252
DEFAULT_TC_BPS = 5.0
COST_LEVELS_BPS = [0.0, 5.0, 10.0, 20.0]
VOL_TARGET_LEVELS = [0.08, 0.10, 0.12]
VOL_TARGET_WINDOW = 63
VOL_TARGET_MAX_LEVERAGE = 1.5
VOL_TARGET_BASE_STRATEGIES = ["Equal Weight", "Minimum Variance", "Risk Parity", "Maximum Sharpe"]
SHRINKAGE_LEVELS = [0.0, 0.25, 0.5, 0.75]
DEFAULT_SHRINKAGE = 0.25
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
    vol_target = pd.read_csv(output_dir / "vol_target_results.csv")
    shrinkage = pd.read_csv(output_dir / "shrinkage_sensitivity.csv")
    benchmark = pd.read_csv(output_dir / "benchmark_relative_metrics.csv")
    stress = pd.read_csv(output_dir / "stress_results.csv")

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

    key_perf = performance.reset_index()[[
        "strategy",
        "cagr",
        "sharpe_ratio",
        "max_drawdown",
        "annualized_volatility",
        "average_turnover",
    ]]
    key_perf = key_perf.sort_values("sharpe_ratio", ascending=False).head(12)

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
    vol_table = vol_target[[
        "target_volatility",
        "strategy",
        "cagr",
        "sharpe_ratio",
        "max_drawdown",
        "annualized_volatility",
        "average_leverage",
    ]]
    shrinkage_table = shrinkage[[
        "shrinkage",
        "strategy",
        "cagr",
        "sharpe_ratio",
        "max_drawdown",
        "annualized_volatility",
    ]]
    benchmark_table = benchmark[[
        "strategy",
        "annualized_alpha_vs_equal_weight",
        "tracking_error_vs_equal_weight",
        "information_ratio_vs_equal_weight",
        "beta_vs_spy",
        "upside_capture_vs_spy",
        "downside_capture_vs_spy",
    ]]
    stress_table = stress[[
        "strategy",
        "max_drawdown",
        "average_drawdown",
        "max_drawdown_duration_days",
        "worst_1_month_return",
        "worst_3_month_return",
        "covid_crash_return_2020_02_19_to_2020_03_23",
        "inflation_rate_shock_return_2022",
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
    best_cagr = performance["cagr"].idxmax()
    lowest_drawdown = performance["max_drawdown"].idxmax()
    best_2022 = stress.set_index("strategy")["inflation_rate_shock_return_2022"].idxmax()

    cost_0 = costs[costs["transaction_cost_bps"] == 0.0].set_index("strategy")
    cost_20 = costs[costs["transaction_cost_bps"] == 20.0].set_index("strategy")
    cost_impact = (cost_20["cagr"] - cost_0["cagr"]).sort_values().iloc[0]

    vol_10 = vol_target[vol_target["target_volatility"] == 0.10].set_index("strategy")
    vol_improvements = []
    for base in VOL_TARGET_BASE_STRATEGIES:
        vt_name = f"{base} Vol Target 10%"
        if vt_name in performance.index and base in performance.index:
            sharpe_delta = performance.loc[vt_name, "sharpe_ratio"] - performance.loc[base, "sharpe_ratio"]
            drawdown_delta = performance.loc[vt_name, "max_drawdown"] - performance.loc[base, "max_drawdown"]
            vol_improvements.append((vt_name, sharpe_delta, drawdown_delta))

    trend_pairs = [
        ("Trend Filtered Equal Weight", "Equal Weight"),
        ("Trend Filtered Minimum Variance", "Minimum Variance"),
        ("Trend Filtered Risk Parity", "Risk Parity"),
    ]
    trend_lines = []
    for trend_strategy, base_strategy in trend_pairs:
        if trend_strategy in performance.index:
            dd_delta = performance.loc[trend_strategy, "max_drawdown"] - performance.loc[base_strategy, "max_drawdown"]
            direction = "improved" if dd_delta > 0 else "worsened"
            trend_lines.append(
                f"{trend_strategy} {direction} max drawdown versus {base_strategy} by {_format_percent(abs(dd_delta))}."
            )

    dual_lines = []
    for strategy in ["Dual Momentum Equal Weight", "Dual Momentum Inverse Vol"]:
        if strategy in performance.index:
            sharpe_delta = performance.loc[strategy, "sharpe_ratio"] - performance.loc["Equal Weight", "sharpe_ratio"]
            dual_lines.append(f"{strategy} Sharpe was {performance.loc[strategy, 'sharpe_ratio']:.2f}, {_format_number(sharpe_delta)} versus Equal Weight.")

    generated = [
        "The tables below are regenerated by `python main.py` from the CSV files in `outputs/`.",
        "",
        "## Key Results",
        "",
        f"- Best Sharpe: {best_sharpe} ({best_sharpe_value:.2f}).",
        f"- Best CAGR: {best_cagr} ({_format_percent(performance.loc[best_cagr, 'cagr'])}).",
        f"- Lowest max drawdown: {lowest_drawdown} ({_format_percent(performance.loc[lowest_drawdown, 'max_drawdown'])}).",
        f"- Best 2022 inflation/rate-shock result: {best_2022} ({_format_percent(stress.set_index('strategy').loc[best_2022, 'inflation_rate_shock_return_2022'])}).",
        f"- Transaction costs: moving from 0 bps to 20 bps reduced the most affected strategy CAGR by {_format_percent(abs(cost_impact))}.",
        "",
        "### Top Strategies by Sharpe",
        "",
        _markdown_table(
            key_perf,
            {
                "cagr": "percent",
                "sharpe_ratio": "number",
                "max_drawdown": "percent",
                "annualized_volatility": "percent",
                "average_turnover": "percent",
            },
        ),
        "",
        "### Interpretation",
        "",
        "- Volatility targeting is applied with lagged 63-day realised volatility and a 1.5x leverage cap. In this run, 10% vol-target variants should be judged against their unscaled base strategies rather than as standalone alpha models.",
        *[
            (
                f"- {name}: Sharpe delta {_format_number(sharpe_delta)}; max drawdown "
                f"{'improved' if drawdown_delta > 0 else 'worsened'} by {_format_percent(abs(drawdown_delta))} "
                "versus the unscaled base strategy."
            )
            for name, sharpe_delta, drawdown_delta in vol_improvements
        ],
        "- Trend filtering uses lagged price signals. It can reduce exposure in prolonged downtrends, but it can also lag reversals and miss recoveries.",
        *[f"- {line}" for line in trend_lines],
        "- Dual momentum is tactical and regime-sensitive. Its results are included as a comparison, not as proof that momentum will persist.",
        *[f"- {line}" for line in dual_lines],
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
        "### Volatility Target Results",
        "",
        _markdown_table(
            vol_table,
            {
                "target_volatility": "percent",
                "cagr": "percent",
                "sharpe_ratio": "number",
                "max_drawdown": "percent",
                "annualized_volatility": "percent",
                "average_leverage": "number",
            },
        ),
        "",
        "### Shrinkage Sensitivity",
        "",
        _markdown_table(
            shrinkage_table,
            {
                "cagr": "percent",
                "sharpe_ratio": "number",
                "max_drawdown": "percent",
                "annualized_volatility": "percent",
            },
        ),
        "",
        "### Benchmark-Relative Metrics",
        "",
        _markdown_table(
            benchmark_table,
            {
                "annualized_alpha_vs_equal_weight": "percent",
                "tracking_error_vs_equal_weight": "percent",
                "information_ratio_vs_equal_weight": "number",
                "beta_vs_spy": "number",
                "upside_capture_vs_spy": "percent",
                "downside_capture_vs_spy": "percent",
            },
        ),
        "",
        "### Stress Results",
        "",
        _markdown_table(
            stress_table,
            {
                "max_drawdown": "percent",
                "average_drawdown": "percent",
                "max_drawdown_duration_days": "integer",
                "worst_1_month_return": "percent",
                "worst_3_month_return": "percent",
                "covid_crash_return_2020_02_19_to_2020_03_23": "percent",
                "inflation_rate_shock_return_2022": "percent",
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


def add_vol_target_strategies(
    returns: pd.DataFrame,
    backtests: dict[str, pd.DataFrame],
    weights: dict[str, pd.DataFrame],
    daily_returns: pd.DataFrame,
    target_volatility: float,
    tc_bps: float,
) -> pd.DataFrame:
    rows = []
    for base_strategy in VOL_TARGET_BASE_STRATEGIES:
        if base_strategy not in weights:
            continue
        strategy_name = f"{base_strategy} Vol Target {target_volatility:.0%}"
        result, scaled_weights, scale = apply_volatility_target(
            returns=returns,
            base_applied_weights=weights[base_strategy],
            target_volatility=target_volatility,
            window=VOL_TARGET_WINDOW,
            max_leverage=VOL_TARGET_MAX_LEVERAGE,
            tc_bps=tc_bps,
        )
        backtests[strategy_name] = result
        weights[strategy_name] = scaled_weights
        daily_returns[strategy_name] = result["net_return"]
        rows.append(
            {
                "target_volatility": target_volatility,
                "strategy": strategy_name,
                "average_leverage": scale[scale > 0.0].mean(),
                "max_leverage": scale.max(),
            }
        )
    return pd.DataFrame(rows)


def build_vol_target_results(
    returns: pd.DataFrame,
    weights: dict[str, pd.DataFrame],
    tc_bps: float,
) -> pd.DataFrame:
    rows = []
    for target_volatility in VOL_TARGET_LEVELS:
        target_backtests = {}
        target_weights = {}
        metadata = []
        for base_strategy in VOL_TARGET_BASE_STRATEGIES:
            result, scaled_weights, scale = apply_volatility_target(
                returns=returns,
                base_applied_weights=weights[base_strategy],
                target_volatility=target_volatility,
                window=VOL_TARGET_WINDOW,
                max_leverage=VOL_TARGET_MAX_LEVERAGE,
                tc_bps=tc_bps,
            )
            strategy_name = f"{base_strategy} Vol Target {target_volatility:.0%}"
            target_backtests[strategy_name] = result
            target_weights[strategy_name] = scaled_weights
            metadata.append(
                {
                    "strategy": strategy_name,
                    "target_volatility": target_volatility,
                    "average_leverage": scale[scale > 0.0].mean(),
                    "max_leverage": scale.max(),
                }
            )

        summary = performance_summary(target_backtests, target_weights).reset_index()
        summary = summary.merge(pd.DataFrame(metadata), on="strategy", how="left")
        rows.append(summary)
    return pd.concat(rows, ignore_index=True)


def build_shrinkage_sensitivity(
    returns: pd.DataFrame,
    prices: pd.DataFrame,
    tc_bps: float,
) -> pd.DataFrame:
    rows = []
    for shrinkage in SHRINKAGE_LEVELS:
        decisions, _ = generate_weight_decisions(
            returns,
            prices=prices,
            estimation_window=ESTIMATION_WINDOW,
            max_weight=MAX_WEIGHT,
            shrinkage=shrinkage,
        )
        backtests, weights, _ = run_all_backtests(returns, decisions, tc_bps=tc_bps)
        selected = {
            strategy: backtests[strategy]
            for strategy in ["Minimum Variance Shrinkage", "Maximum Sharpe Shrinkage", "Risk Parity Shrinkage"]
        }
        selected_weights = {strategy: weights[strategy] for strategy in selected}
        summary = performance_summary(selected, selected_weights).reset_index()
        summary.insert(0, "shrinkage", shrinkage)
        rows.append(summary)
    return pd.concat(rows, ignore_index=True)


def run_backtests_with_default_vol_targets(
    returns: pd.DataFrame,
    decisions: dict[str, pd.DataFrame],
    tc_bps: float,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], pd.DataFrame]:
    backtests, weights, daily_returns = run_all_backtests(returns, decisions, tc_bps=tc_bps)
    add_vol_target_strategies(
        returns=returns,
        backtests=backtests,
        weights=weights,
        daily_returns=daily_returns,
        target_volatility=0.10,
        tc_bps=tc_bps,
    )
    return backtests, weights, daily_returns


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    (OUTPUT_DIR / ".gitkeep").touch()

    tickers = list(ETF_UNIVERSE)
    prices = download_prices(tickers=tickers, start=START_DATE, end=END_DATE)
    returns = calculate_returns(prices).loc[: "2024-12-31"].dropna(how="any")

    decisions, risk_records = generate_weight_decisions(
        returns,
        prices=prices.loc[returns.index],
        estimation_window=ESTIMATION_WINDOW,
        max_weight=MAX_WEIGHT,
        shrinkage=DEFAULT_SHRINKAGE,
    )
    backtests, weights, daily_returns = run_backtests_with_default_vol_targets(
        returns,
        decisions,
        tc_bps=DEFAULT_TC_BPS,
    )

    performance = performance_summary(backtests, weights)
    subperiods = subperiod_summary(daily_returns, SUBPERIODS)
    average_weights = summarize_average_weights(weights)
    risk_contribution = summarize_risk_contributions(risk_records)
    vol_target_results = build_vol_target_results(returns, weights, tc_bps=DEFAULT_TC_BPS)
    shrinkage_sensitivity = build_shrinkage_sensitivity(returns, prices.loc[returns.index], tc_bps=DEFAULT_TC_BPS)
    benchmark_metrics = benchmark_relative_metrics(daily_returns, returns)
    stress = stress_results(daily_returns)

    cost_rows = []
    for cost_bps in COST_LEVELS_BPS:
        cost_backtests, cost_weights, _ = run_backtests_with_default_vol_targets(
            returns,
            decisions,
            tc_bps=cost_bps,
        )
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
    vol_target_results.to_csv(OUTPUT_DIR / "vol_target_results.csv", index=False)
    shrinkage_sensitivity.to_csv(OUTPUT_DIR / "shrinkage_sensitivity.csv", index=False)
    benchmark_metrics.to_csv(OUTPUT_DIR / "benchmark_relative_metrics.csv", index=False)
    stress.to_csv(OUTPUT_DIR / "stress_results.csv", index=False)
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
    plot_equity_curves(daily_returns, OUTPUT_DIR / "equity_curves_extended.png")
    plot_drawdowns(daily_returns, OUTPUT_DIR / "drawdowns_extended.png")
    plot_strategy_group(
        daily_returns,
        [
            "Equal Weight",
            "Equal Weight Vol Target 10%",
            "Minimum Variance",
            "Minimum Variance Vol Target 10%",
            "Risk Parity",
            "Risk Parity Vol Target 10%",
            "Maximum Sharpe",
            "Maximum Sharpe Vol Target 10%",
        ],
        OUTPUT_DIR / "vol_target_comparison.png",
        "Volatility Target Comparison",
    )
    plot_strategy_group(
        daily_returns,
        [
            "Equal Weight",
            "Trend Filtered Equal Weight",
            "Minimum Variance",
            "Trend Filtered Minimum Variance",
            "Risk Parity",
            "Trend Filtered Risk Parity",
        ],
        OUTPUT_DIR / "trend_filter_comparison.png",
        "Trend Filter Comparison",
    )
    plot_strategy_group(
        daily_returns,
        ["Equal Weight", "Dual Momentum Equal Weight", "Dual Momentum Inverse Vol", "SPY"],
        OUTPUT_DIR / "dual_momentum_comparison.png",
        "Dual Momentum Comparison",
    )
    plot_stress_period_comparison(stress, OUTPUT_DIR / "stress_period_comparison.png")

    update_readme_results(OUTPUT_DIR)

    print("Generated allocation engine outputs in outputs/.")
    print(performance.to_string(float_format=lambda value: f"{value:0.4f}"))


if __name__ == "__main__":
    main()
