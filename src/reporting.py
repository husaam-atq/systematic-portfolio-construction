from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


DISPLAY_METHODS = [
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

BASELINE_DISPLAY_METHODS = [
    "Equal Weight",
    "Minimum Variance",
    "Maximum Sharpe",
    "Dual Momentum Equal Weight",
    "Maximum Sharpe Vol Target 10%",
]


def _percent(value: object) -> str:
    return "" if pd.isna(value) else f"{float(value):.2%}"


def _number(value: object) -> str:
    return "" if pd.isna(value) else f"{float(value):.2f}"


def markdown_table(frame: pd.DataFrame, formats: dict[str, str]) -> str:
    formatted = frame.copy()
    for column, style in formats.items():
        if column not in formatted:
            continue
        if style == "percent":
            formatted[column] = formatted[column].map(_percent)
        elif style == "number":
            formatted[column] = formatted[column].map(_number)
        elif style == "integer":
            formatted[column] = formatted[column].map(lambda value: "" if pd.isna(value) else str(int(value)))
    columns = [str(column) for column in formatted.columns]
    rows = [[str(value) for value in row] for row in formatted.to_numpy()]
    widths = [max([len(columns[index]), *[len(row[index]) for row in rows]]) for index in range(len(columns))]
    header = "| " + " | ".join(value.ljust(widths[index]) for index, value in enumerate(columns)) + " |"
    divider = "| " + " | ".join("-" * widths[index] for index in range(len(columns))) + " |"
    body = [
        "| " + " | ".join(value.ljust(widths[index]) for index, value in enumerate(row)) + " |"
        for row in rows
    ]
    return "\n".join([header, divider, *body])


def _selected_performance(performance: pd.DataFrame) -> pd.DataFrame:
    methods = [method for method in DISPLAY_METHODS if method in performance.index]
    return performance.loc[methods].reset_index()[
        [
            "strategy",
            "cagr",
            "sharpe_ratio",
            "annualized_volatility",
            "max_drawdown",
            "average_monthly_turnover",
            "effective_assets_by_weight",
        ]
    ]


def _baseline_table(performance: pd.DataFrame | None) -> str:
    if performance is None:
        return "The preserved baseline table is available in `reports/baseline_report.md`."
    methods = [method for method in BASELINE_DISPLAY_METHODS if method in performance.index]
    selected = performance.loc[methods].reset_index()[
        ["strategy", "cagr", "sharpe_ratio", "max_drawdown"]
    ]
    return markdown_table(
        selected,
        {"cagr": "percent", "sharpe_ratio": "number", "max_drawdown": "percent"},
    )


def _development_findings(
    performance: pd.DataFrame,
    risk_summary: pd.DataFrame | None,
    uncertainty: pd.DataFrame | None,
    stability: pd.DataFrame | None,
) -> str:
    best_sharpe = performance["sharpe_ratio"].idxmax()
    best_cagr = performance["cagr"].idxmax()
    shallowest = performance["max_drawdown"].idxmax()
    if best_sharpe == best_cagr:
        benchmark_finding = (
            f"- **Simple benchmark:** {best_sharpe} led both development Sharpe "
            f"({_number(performance.loc[best_sharpe, 'sharpe_ratio'])}) and CAGR "
            f"({_percent(performance.loc[best_cagr, 'cagr'])})."
        )
    else:
        benchmark_finding = (
            f"- **Simple benchmark:** {best_sharpe} had the highest development Sharpe "
            f"({_number(performance.loc[best_sharpe, 'sharpe_ratio'])}), while {best_cagr} had the highest CAGR "
            f"({_percent(performance.loc[best_cagr, 'cagr'])})."
        )
    benchmark_finding += (
        f" The Sharpe-leading portfolio had a {_percent(performance.loc[best_sharpe, 'max_drawdown'])} maximum drawdown, "
        "so this is evidence, not a universal allocation recommendation."
    )
    lines = [
        benchmark_finding,
        f"- **Risk reduction:** Minimum Variance reduced annualised volatility from "
        f"{_percent(performance.loc['Equal Weight', 'annualized_volatility'])} to "
        f"{_percent(performance.loc['Minimum Variance', 'annualized_volatility'])}, but its CAGR fell from "
        f"{_percent(performance.loc['Equal Weight', 'cagr'])} to {_percent(performance.loc['Minimum Variance', 'cagr'])}.",
        f"- **Drawdown trade-off:** {shallowest} had the shallowest maximum drawdown "
        f"({_percent(performance.loc[shallowest, 'max_drawdown'])}) but only "
        f"{_percent(performance.loc[shallowest, 'cagr'])} CAGR. Trend filtering behaved as insurance rather than free alpha.",
        f"- **Shrinkage and HRP:** 25% diagonal shrinkage changed Maximum Sharpe CAGR from "
        f"{_percent(performance.loc['Maximum Sharpe', 'cagr'])} to "
        f"{_percent(performance.loc['Maximum Sharpe Shrinkage', 'cagr'])} and monthly turnover from "
        f"{_percent(performance.loc['Maximum Sharpe', 'average_monthly_turnover'])} to "
        f"{_percent(performance.loc['Maximum Sharpe Shrinkage', 'average_monthly_turnover'])}. HRP did not improve the development Sharpe over Equal Weight.",
        f"- **Volatility targeting:** the 10% Maximum Sharpe overlay changed Sharpe from "
        f"{_number(performance.loc['Maximum Sharpe', 'sharpe_ratio'])} to "
        f"{_number(performance.loc['Maximum Sharpe Vol Target 10%', 'sharpe_ratio'])}, while drawdown moved from "
        f"{_percent(performance.loc['Maximum Sharpe', 'max_drawdown'])} to "
        f"{_percent(performance.loc['Maximum Sharpe Vol Target 10%', 'max_drawdown'])} and turnover rose to "
        f"{_percent(performance.loc['Maximum Sharpe Vol Target 10%', 'average_monthly_turnover'])}. Financing and leverage turnover erase any claim of mechanically improved risk.",
        f"- **Tactical overlay:** Dual Momentum Equal Weight raised development CAGR to "
        f"{_percent(performance.loc['Dual Momentum Equal Weight', 'cagr'])}, but monthly turnover was "
        f"{_percent(performance.loc['Dual Momentum Equal Weight', 'average_monthly_turnover'])} and its bootstrap Sharpe advantage was not decisive.",
    ]
    if risk_summary is not None:
        risk = risk_summary.groupby("strategy").first()
        if "Risk Parity" in risk.index:
            lines.append(
                f"- **Risk allocation:** capped Risk Parity averaged "
                f"{_number(risk.loc['Risk Parity', 'effective_assets_by_risk'])} effective risk contributors versus "
                f"{_number(risk.loc['Risk Parity', 'effective_assets_by_weight'])} effective assets by weight. Exact equality is sometimes infeasible when the 40% cap binds."
            )
    if stability is not None and "Maximum Sharpe" in stability.set_index("strategy").index:
        row = stability.set_index("strategy").loc["Maximum Sharpe"]
        lines.append(
            f"- **Estimator instability:** Maximum Sharpe solved successfully on {_percent(row['solver_success_rate'])} of monthly decisions but averaged "
            f"{_percent(row['average_monthly_turnover'])} monthly turnover and regularly hit bounds; it remains a diagnostic method, not the selected representative."
        )
    if uncertainty is not None:
        intervals = uncertainty[uncertainty["metric"] == "sharpe_difference_vs_equal_weight"]
        decisive = intervals[(intervals["lower_95"] > 0.0) | (intervals["upper_95"] < 0.0)]["strategy"].tolist()
        lines.append(
            "- **Uncertainty:** only "
            + (", ".join(decisive) if decisive else "no reported method")
            + " had a 95% moving-block-bootstrap Sharpe-difference interval excluding zero versus Equal Weight. These intervals remain conditional on one historical ETF sample."
        )
    return "\n".join(lines)


def _confirmation_findings(performance: pd.DataFrame) -> str:
    best_sharpe = performance["sharpe_ratio"].idxmax()
    best_cagr = performance["cagr"].idxmax()
    shallowest = performance["max_drawdown"].idxmax()
    equal_weight = performance.loc["Equal Weight"]
    return "\n".join(
        [
            f"- Highest confirmation Sharpe: {best_sharpe} ({_number(performance.loc[best_sharpe, 'sharpe_ratio'])}).",
            f"- Highest confirmation CAGR: {best_cagr} ({_percent(performance.loc[best_cagr, 'cagr'])}).",
            f"- Shallowest confirmation maximum drawdown: {shallowest} ({_percent(performance.loc[shallowest, 'max_drawdown'])}).",
            f"- Equal Weight recorded {_percent(equal_weight['cagr'])} CAGR, {_number(equal_weight['sharpe_ratio'])} Sharpe and {_percent(equal_weight['max_drawdown'])} maximum drawdown over the same short interval.",
        ]
    )


def write_development_report(
    path: str | Path,
    performance: pd.DataFrame,
    segments: pd.DataFrame,
    uncertainty: pd.DataFrame,
    stability: pd.DataFrame,
    risk_parity_check: pd.DataFrame,
    sensitivity: pd.DataFrame,
    evaluation_start: str | pd.Timestamp,
) -> None:
    selected = _selected_performance(performance)
    best_sharpe = performance["sharpe_ratio"].idxmax()
    best_drawdown = performance["max_drawdown"].idxmax()
    max_sharpe_stability = stability[stability["strategy"].str.contains("Maximum Sharpe", regex=False)]
    uncertainty_sharpe = uncertainty[uncertainty["metric"] == "sharpe_difference_vs_equal_weight"]
    indistinguishable = uncertainty_sharpe[
        (uncertainty_sharpe["lower_95"] <= 0.0) & (uncertainty_sharpe["upper_95"] >= 0.0)
    ]["strategy"].tolist()

    shrinkage_comparison = []
    for base, shrunk in [
        ("Minimum Variance", "Minimum Variance Shrinkage"),
        ("Maximum Sharpe", "Maximum Sharpe Shrinkage"),
        ("Risk Parity", "Risk Parity Shrinkage"),
    ]:
        if base in performance.index and shrunk in performance.index:
            shrinkage_comparison.append(
                f"- {shrunk}: Sharpe change {_number(performance.loc[shrunk, 'sharpe_ratio'] - performance.loc[base, 'sharpe_ratio'])}; "
                f"turnover change {_percent(performance.loc[shrunk, 'average_monthly_turnover'] - performance.loc[base, 'average_monthly_turnover'])}."
            )

    text = f"""# Development Results: 2010-2024

## Scope

These are development-period walk-forward results under the methodology documented in `reports/research_design.md`. They are not confirmation results and were not used to tune the final holdout period. After causal estimator and volatility-overlay warm-ups, the common comparison starts {pd.Timestamp(evaluation_start).date()}.

Reported Sharpe and Sortino ratios use contemporaneous SHY returns as the cash hurdle. CAGR, volatility and drawdown remain total-return measures.
All strategies in the headline table use the same evaluation start after the longest required warm-up. Sensitivity comparisons likewise align competing parameter values to a common start date.

## Representative Performance

{markdown_table(selected, {'cagr': 'percent', 'sharpe_ratio': 'number', 'annualized_volatility': 'percent', 'max_drawdown': 'percent', 'average_monthly_turnover': 'percent', 'effective_assets_by_weight': 'number'})}

## What Survived Stronger Controls

- The highest development Sharpe was {best_sharpe} ({performance.loc[best_sharpe, 'sharpe_ratio']:.2f}), but that ranking is not the research decision rule.
- The shallowest development drawdown was {best_drawdown} ({performance.loc[best_drawdown, 'max_drawdown']:.2%}).
- Methods whose 95% moving-block-bootstrap interval for the Sharpe difference versus Equal Weight included zero: {', '.join(indistinguishable) if indistinguishable else 'none in the reported set'}.
- Equal Weight and the fixed 60/40 benchmark remain essential references because they avoid estimated expected returns and have transparent implementation.
- Maximum Sharpe remains an audit method rather than an endorsed representative portfolio: its sample-mean inputs, bound frequency, turnover and perturbation sensitivity are reported explicitly.

## Covariance Shrinkage

{chr(10).join(shrinkage_comparison)}

Shrinkage is treated as estimator regularisation, not a guarantee of improved realised performance. The full 0%/25%/50%/75% sensitivity is retained in `outputs/parameter_sensitivity.csv`.

## HRP

HRP was added because it avoids expected-return estimation and direct mean-variance inversion. Its evidence is judged on stability, concentration, turnover and segment behaviour rather than whether it tops the full-sample leaderboard.

## Risk Parity Verification

{markdown_table(risk_parity_check, {'mean_absolute_risk_contribution_error': 'number', 'maximum_risk_contribution_error': 'number', 'observations': 'integer'}) if not risk_parity_check.empty else 'No Risk Parity verification rows were produced.'}

These diagnostics apply to canonical Risk Parity variants. Exact equality can be infeasible when the 40% long-only cap binds; trend-filtered Risk Parity and HRP are not asserted to be equal-risk-contribution portfolios.

## Estimator Stability

{markdown_table(max_sharpe_stability, {'solver_success_rate': 'percent', 'median_covariance_condition_number': 'number', 'maximum_covariance_condition_number': 'number', 'average_bound_count': 'number', 'average_weight_dispersion': 'number', 'average_effective_assets': 'number', 'average_max_weight': 'percent', 'average_l1_weight_change': 'number', 'worst_l1_weight_change': 'number', 'average_monthly_turnover': 'percent', 'effective_assets_by_weight': 'number'}) if not max_sharpe_stability.empty else 'No Maximum Sharpe stability rows were produced.'}

## Segment Evidence

Full segment results are retained in `outputs/walk_forward_segments.csv`. No single full-sample leaderboard is treated as the primary conclusion.

## Parameter Robustness

The development run reports estimation-window, rebalance-frequency, covariance-shrinkage, asset-cap, transaction-cost, volatility-target, tactical-lookback, expected-return-estimator and delayed-execution sensitivity. The matrix is deliberately small and economically motivated rather than an optimisation grid.

## Development Decision

The frozen confirmation set should include simple benchmarks and materially distinct methods: Equal Weight, Traditional 60/40, Inverse Volatility, Minimum Variance, Minimum Variance Shrinkage, Risk Parity, Risk Parity Shrinkage, HRP, Maximum Sharpe, Maximum Sharpe Shrinkage, Dual Momentum Equal Weight and the 10% Maximum Sharpe volatility-target overlay. Inclusion is for comparison, not endorsement. No method is selected solely because it has the best development CAGR or Sharpe.
"""
    Path(path).write_text(text, encoding="utf-8")


def write_experiment_log(path: str | Path, sensitivity: pd.DataFrame, performance: pd.DataFrame) -> None:
    lines = [
        "# Experiment Log",
        "",
        "All experiments below were specified before confirmation. Weak and negative results are retained.",
        "",
        "## Main Development Methods",
        "",
        markdown_table(
            _selected_performance(performance),
            {
                "cagr": "percent",
                "sharpe_ratio": "number",
                "annualized_volatility": "percent",
                "max_drawdown": "percent",
                "average_monthly_turnover": "percent",
                "effective_assets_by_weight": "number",
            },
        ),
        "",
        "## Sensitivity Families",
        "",
    ]
    for dimension in sensitivity["dimension"].drop_duplicates():
        subset = sensitivity[sensitivity["dimension"] == dimension]
        best = subset.loc[subset["sharpe_ratio"].idxmax()]
        worst = subset.loc[subset["sharpe_ratio"].idxmin()]
        lines.extend(
            [
                f"### {dimension}",
                "",
                f"- Highest reported Sharpe in this diagnostic family: {best['strategy']} at `{best['value']}` ({best['sharpe_ratio']:.2f}).",
                f"- Lowest reported Sharpe in this diagnostic family: {worst['strategy']} at `{worst['value']}` ({worst['sharpe_ratio']:.2f}).",
                "- This comparison is descriptive; the best value was not selected for confirmation from this ranking.",
                "",
            ]
        )
    lines.append("The complete experiment matrix is available in `outputs/parameter_sensitivity.csv`.")
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def write_limitations(path: str | Path) -> None:
    Path(path).write_text(
        """# Limitations

- The nine-ETF universe is small, selected ex post and subject to survivorship and product-selection bias.
- Adjusted ETF histories are investable proxies, not complete asset-class histories, and vendor revisions can change past values.
- SHY is both an investable asset and the cash-rate proxy; this is transparent but simplified.
- The financing model uses SHY plus a fixed 50 bps annual spread. Real borrowing rates, margin requirements and availability vary.
- Fixed-bps transaction costs omit bid-ask variation, taxes, market impact and capacity.
- Maximum Sharpe depends on noisy expected-return estimates even when covariance is shrunk.
- Risk contribution is ex ante and covariance-model dependent; realised crisis correlations can differ sharply.
- HRP depends on the clustering rule and can change when correlations are close.
- Trend and dual momentum are regime dependent and their alternative lookbacks are not independent experiments.
- Volatility targeting can lever after calm periods and react slowly to abrupt volatility jumps.
- Moving-block-bootstrap intervals depend on block length and do not turn one historical path into independent evidence.
- Fixed stress windows are economically motivated but do not span every relevant market event.
- The 2025+ confirmation period is short and can only provide limited fresh evidence.
- Past diversification and benchmark-relative performance may not persist.
""",
        encoding="utf-8",
    )


def write_confirmation_report(
    path: str | Path,
    performance: pd.DataFrame | None = None,
    freeze_hash: str | None = None,
    confirmation_start: str | pd.Timestamp | None = None,
    confirmation_end: str | pd.Timestamp | None = None,
) -> None:
    if performance is None:
        text = """# Confirmation Results

Confirmation has not been run. The 2025+ period remains uninspected until the protocol is frozen, hashed and committed.
"""
    else:
        selected = _selected_performance(performance)
        findings = _confirmation_findings(performance)
        text = f"""# Frozen Confirmation Results: 2025+

The confirmation was run once under frozen protocol SHA-256 `{freeze_hash}`. Methodology was not changed after viewing these results.
The observed holdout runs from {pd.Timestamp(confirmation_start).date()} through {pd.Timestamp(confirmation_end).date()}.

{markdown_table(selected, {'cagr': 'percent', 'sharpe_ratio': 'number', 'annualized_volatility': 'percent', 'max_drawdown': 'percent', 'average_monthly_turnover': 'percent', 'effective_assets_by_weight': 'number'})}

## Observed Results

{findings}

This short confirmation period is evidence about recent behaviour, not a new optimisation sample. No method is declared a winner from this interval alone.
"""
    Path(path).write_text(text, encoding="utf-8")


def write_readme(
    path: str | Path,
    development_performance: pd.DataFrame,
    confirmation_performance: pd.DataFrame | None = None,
    freeze_hash: str | None = None,
    development_evaluation_start: str | pd.Timestamp | None = None,
    confirmation_end: str | pd.Timestamp | None = None,
    baseline_performance: pd.DataFrame | None = None,
    development_risk_summary: pd.DataFrame | None = None,
    development_uncertainty: pd.DataFrame | None = None,
    development_stability: pd.DataFrame | None = None,
) -> None:
    development = _selected_performance(development_performance)
    confirmation_section = (
        "Confirmation remains sealed until the frozen protocol is committed."
        if confirmation_performance is None
        else (
            f"Confirmation through {pd.Timestamp(confirmation_end).date()}:\n\n"
            + markdown_table(
                _selected_performance(confirmation_performance),
                {
                    "cagr": "percent",
                    "sharpe_ratio": "number",
                    "annualized_volatility": "percent",
                    "max_drawdown": "percent",
                    "average_monthly_turnover": "percent",
                    "effective_assets_by_weight": "number",
                },
            )
            + "\n\n"
            + _confirmation_findings(confirmation_performance)
        )
    )
    freeze_line = freeze_hash or "pending"
    findings = _development_findings(
        development_performance,
        development_risk_summary,
        development_uncertainty,
        development_stability,
    )
    text = f"""# Systematic Portfolio Construction & Risk Allocation Research Framework

Which portfolio-construction methods remain useful once estimation error, concentration, costs and changing market regimes are taken seriously?

This repository is a research framework, not a strategy promising superior returns. It compares static allocation, risk-management overlays and tactical methods under causal walk-forward estimation, drifted holdings, transaction costs, cash yield, financing costs, stress tests and dependence-aware uncertainty.

Sharpe and Sortino ratios are calculated from returns in excess of the contemporaneous SHY cash proxy; return and drawdown statistics use total net portfolio returns.

## Research Integrity

- The original engine and generated evidence are archived under `archive/baseline_48bcbe9/`.
- Development research is fixed to 2010-2024.
- The 2025+ protocol is frozen separately before confirmation and identified by SHA-256 `{freeze_line}`.
- Weak and negative experiments remain in `reports/experiment_log.md` and `outputs/parameter_sensitivity.csv`.
- No method is selected solely on CAGR or Sharpe.

## Original Baseline

{_baseline_table(baseline_performance)}

The original engine forward-filled target weights instead of drifting holdings, measured turnover between abstract targets, assumed zero return on cash, charged no financing spread above 1x, and used a zero cash hurdle for Sharpe. The archive preserves those outputs; they are not targets for the upgraded framework.

## Development Evidence

The common post-warm-up evaluation begins {pd.Timestamp(development_evaluation_start).date() if development_evaluation_start is not None else 'after all methods are active'}.

{markdown_table(development, {'cagr': 'percent', 'sharpe_ratio': 'number', 'annualized_volatility': 'percent', 'max_drawdown': 'percent', 'average_monthly_turnover': 'percent', 'effective_assets_by_weight': 'number'})}

The table is generated from `outputs/development_performance.csv`. Differences are interpreted alongside walk-forward segments, turnover, concentration, estimator stability, risk contribution and moving-block-bootstrap intervals. A simple method can be more credible than a historically superior optimiser.

## Key Findings

{findings}

## Fresh Confirmation

{confirmation_section}

## Methods

Static risk allocation:

- Equal Weight and a fixed SPY/IEF 60/40 benchmark
- Inverse Volatility
- Minimum Variance and diagonal-covariance-shrinkage Minimum Variance
- Maximum Sharpe and shrinkage Maximum Sharpe, retained with explicit stability diagnostics
- Risk Parity / Equal Risk Contribution and its shrinkage variant
- Hierarchical Risk Parity

Risk-management overlays:

- 8%, 10% and 12% volatility targets with a 63-day trailing estimator and 1.5x cap
- Positive cash earns SHY; negative cash pays SHY plus a 50 bps annual financing spread

Tactical methods, reported separately:

- Trend-filtered Equal Weight, Minimum Variance and Risk Parity
- Dual Momentum Equal Weight and Inverse Volatility

## Economic Simulation

Targets formed at close t first earn return t+1. Holdings drift between rebalances. Turnover is measured from drifted pre-trade holdings to the new target, and 5 bps one-way costs are charged on actual ETF trades. The same simulator is used for methods and benchmarks.

Headline comparisons begin only when every reported method has completed its warm-up. Parameter-sensitivity families use a common evaluation start within each family.

## Universe

| ETF | Role |
|---|---|
| SPY | US equities |
| EFA | Developed ex-US equities |
| EEM | Emerging markets equities |
| TLT | Long-term US Treasuries |
| IEF | Intermediate US Treasuries |
| SHY | Short-duration US Treasuries and public cash proxy |
| GLD | Gold |
| VNQ | REITs |
| DBC | Commodities |

## Reproduce

```bash
pip install -r requirements.txt
python main.py
python -m pytest -q
```

`main.py` is development-only and cannot request 2025+ data. After the protocol artifact is frozen and hash-verified, `python confirmation.py` runs the fixed confirmation workflow.

## Reports

- [Baseline reproduction](reports/baseline_report.md)
- [Research design and audit](reports/research_design.md)
- [Development results](reports/development_results.md)
- [Confirmation results](reports/confirmation_results.md)
- [Experiment log](reports/experiment_log.md)
- [Limitations](reports/limitations.md)

## Figures

![Development equity](outputs/figures/development_equity.png)

![Development drawdown](outputs/figures/development_drawdown.png)

![Average weights](outputs/figures/average_weights.png)

![Risk contributions](outputs/figures/risk_contributions.png)

![Effective diversification](outputs/figures/effective_diversification.png)

![Rolling volatility](outputs/figures/rolling_portfolio_volatility.png)

![Rolling allocation](outputs/figures/rolling_asset_allocation.png)

![Cost sensitivity](outputs/figures/turnover_cost_sensitivity.png)

![Estimation-window sensitivity](outputs/figures/estimation_window_sensitivity.png)

![Stress comparison](outputs/figures/stress_period_comparison.png)

![Benchmark-relative performance](outputs/figures/benchmark_relative_performance.png)

## Limitations

ETF selection matters; the universe is small and survivorship-biased. SHY is a simplified cash proxy. Financing and transaction costs are stylised. Optimised weights depend on noisy estimates. Tactical signals and volatility forecasts are regime dependent. Bootstrap intervals and a short confirmation period do not eliminate uncertainty. See `reports/limitations.md` for the full discussion.
"""
    Path(path).write_text(text, encoding="utf-8")
