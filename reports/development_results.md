# Development Results: 2010-2024

## Scope

These are development-period walk-forward results under the methodology documented in `reports/research_design.md`. They are not confirmation results and were not used to tune the final holdout period. After causal estimator and volatility-overlay warm-ups, the common comparison starts 2011-05-02.

Reported Sharpe and Sortino ratios use contemporaneous SHY returns as the cash hurdle. CAGR, volatility and drawdown remain total-return measures.
All strategies in the headline table use the same evaluation start after the longest required warm-up. Sensitivity comparisons likewise align competing parameter values to a common start date.

## Representative Performance

| strategy                      | cagr  | sharpe_ratio | annualized_volatility | max_drawdown | average_monthly_turnover | effective_assets_by_weight |
| ----------------------------- | ----- | ------------ | --------------------- | ------------ | ------------------------ | -------------------------- |
| Equal Weight                  | 4.17% | 0.39         | 8.85%                 | -18.54%      | 3.10%                    | 8.99                       |
| Traditional 60/40             | 8.93% | 0.82         | 9.79%                 | -21.29%      | 2.26%                    | 1.92                       |
| Inverse Volatility            | 2.83% | 0.40         | 5.08%                 | -14.74%      | 3.01%                    | 4.65                       |
| Minimum Variance              | 2.66% | 0.55         | 3.66%                 | -13.10%      | 4.60%                    | 3.00                       |
| Minimum Variance Shrinkage    | 2.55% | 0.51         | 3.69%                 | -13.14%      | 4.50%                    | 3.07                       |
| Risk Parity                   | 2.65% | 0.39         | 4.71%                 | -14.48%      | 3.37%                    | 4.50                       |
| Risk Parity Shrinkage         | 2.66% | 0.39         | 4.74%                 | -14.53%      | 3.26%                    | 4.52                       |
| Hierarchical Risk Parity      | 2.02% | 0.28         | 4.38%                 | -15.27%      | 6.25%                    | 3.71                       |
| Maximum Sharpe                | 4.84% | 0.52         | 7.90%                 | -18.94%      | 39.55%                   | 3.27                       |
| Maximum Sharpe Shrinkage      | 4.91% | 0.52         | 8.04%                 | -18.54%      | 34.92%                   | 3.48                       |
| Dual Momentum Equal Weight    | 6.67% | 0.54         | 11.31%                | -18.71%      | 34.33%                   | 2.93                       |
| Maximum Sharpe Vol Target 10% | 5.62% | 0.53         | 9.25%                 | -20.63%      | 65.10%                   | 3.27                       |

## What Survived Stronger Controls

- The highest development Sharpe was Traditional 60/40 (0.82), but that ranking is not the research decision rule.
- The shallowest development drawdown was Trend Filtered Risk Parity (-7.44%).
- Methods whose 95% moving-block-bootstrap interval for the Sharpe difference versus Equal Weight included zero: Equal Weight, Inverse Volatility, Minimum Variance, Minimum Variance Shrinkage, Maximum Sharpe, Maximum Sharpe Shrinkage, Risk Parity, Risk Parity Shrinkage, Hierarchical Risk Parity, Trend Filtered Equal Weight, Trend Filtered Minimum Variance, Trend Filtered Risk Parity, Dual Momentum Equal Weight, Dual Momentum Inverse Volatility, Equal Weight Vol Target 10%, Minimum Variance Vol Target 10%, Maximum Sharpe Vol Target 10%, Risk Parity Vol Target 10%.
- Equal Weight and the fixed 60/40 benchmark remain essential references because they avoid estimated expected returns and have transparent implementation.
- Maximum Sharpe remains an audit method rather than an endorsed representative portfolio: its sample-mean inputs, bound frequency, turnover and perturbation sensitivity are reported explicitly.

## Covariance Shrinkage

- Minimum Variance Shrinkage: Sharpe change -0.04; turnover change -0.10%.
- Maximum Sharpe Shrinkage: Sharpe change 0.00; turnover change -4.63%.
- Risk Parity Shrinkage: Sharpe change -0.00; turnover change -0.10%.

Shrinkage is treated as estimator regularisation, not a guarantee of improved realised performance. The full 0%/25%/50%/75% sensitivity is retained in `outputs/parameter_sensitivity.csv`.

## HRP

HRP was added because it avoids expected-return estimation and direct mean-variance inversion. Its evidence is judged on stability, concentration, turnover and segment behaviour rather than whether it tops the full-sample leaderboard.

## Risk Parity Verification

| strategy                   | mean_absolute_risk_contribution_error | maximum_risk_contribution_error | observations |
| -------------------------- | ------------------------------------- | ------------------------------- | ------------ |
| Risk Parity                | 0.01                                  | 0.10                            | 164          |
| Risk Parity Shrinkage      | 0.01                                  | 0.10                            | 164          |
| Risk Parity Vol Target 10% | 0.01                                  | 0.10                            | 164          |

These diagnostics apply to canonical Risk Parity variants. Exact equality can be infeasible when the 40% long-only cap binds; trend-filtered Risk Parity and HRP are not asserted to be equal-risk-contribution portfolios.

## Estimator Stability

| strategy                 | solver_success_rate | median_covariance_condition_number | maximum_covariance_condition_number | average_bound_count | average_weight_dispersion | average_effective_assets | average_max_weight | average_l1_weight_change | worst_l1_weight_change | average_monthly_turnover | effective_assets_by_weight |
| ------------------------ | ------------------- | ---------------------------------- | ----------------------------------- | ------------------- | ------------------------- | ------------------------ | ------------------ | ------------------------ | ---------------------- | ------------------------ | -------------------------- |
| Maximum Sharpe           | 98.21%              | 4586.20                            | 23333.99                            | 1.24                | 0.15                      | 3.28                     | 39.27%             | 0.01                     | 0.06                   | 39.55%                   | 3.27                       |
| Maximum Sharpe Shrinkage | 98.81%              | 1483.61                            | 12536.27                            | 1.03                | 0.14                      | 3.49                     | 38.62%             | 0.01                     | 0.06                   | 34.92%                   | 3.48                       |

## Segment Evidence

Full segment results are retained in `outputs/walk_forward_segments.csv`. No single full-sample leaderboard is treated as the primary conclusion.

## Parameter Robustness

The development run reports estimation-window, rebalance-frequency, covariance-shrinkage, asset-cap, transaction-cost, volatility-target, tactical-lookback, expected-return-estimator and delayed-execution sensitivity. The matrix is deliberately small and economically motivated rather than an optimisation grid.

## Development Decision

The frozen confirmation set should include simple benchmarks and materially distinct methods: Equal Weight, Traditional 60/40, Inverse Volatility, Minimum Variance, Minimum Variance Shrinkage, Risk Parity, Risk Parity Shrinkage, HRP, Maximum Sharpe, Maximum Sharpe Shrinkage, Dual Momentum Equal Weight and the 10% Maximum Sharpe volatility-target overlay. Inclusion is for comparison, not endorsement. No method is selected solely because it has the best development CAGR or Sharpe.
