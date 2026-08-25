# Experiment Log

All experiments below were specified before confirmation. Weak and negative results are retained.

## Main Development Methods

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

## Sensitivity Families

### estimation_window

- Highest reported Sharpe in this diagnostic family: Maximum Sharpe at `504` (0.68).
- Lowest reported Sharpe in this diagnostic family: Hierarchical Risk Parity at `252` (0.16).
- This comparison is descriptive; the best value was not selected for confirmation from this ranking.

### rebalance_frequency

- Highest reported Sharpe in this diagnostic family: Minimum Variance at `monthly` (0.58).
- Lowest reported Sharpe in this diagnostic family: Hierarchical Risk Parity at `quarterly` (0.30).
- This comparison is descriptive; the best value was not selected for confirmation from this ranking.

### covariance_shrinkage

- Highest reported Sharpe in this diagnostic family: Minimum Variance Shrinkage at `0.0` (0.58).
- Lowest reported Sharpe in this diagnostic family: Risk Parity Shrinkage at `0.5` (0.44).
- This comparison is descriptive; the best value was not selected for confirmation from this ranking.

### maximum_asset_weight

- Highest reported Sharpe in this diagnostic family: Minimum Variance at `0.3` (0.65).
- Lowest reported Sharpe in this diagnostic family: Hierarchical Risk Parity at `0.6` (0.33).
- This comparison is descriptive; the best value was not selected for confirmation from this ranking.

### transaction_cost_bps

- Highest reported Sharpe in this diagnostic family: Minimum Variance at `0.0` (0.59).
- Lowest reported Sharpe in this diagnostic family: Hierarchical Risk Parity at `20.0` (0.30).
- This comparison is descriptive; the best value was not selected for confirmation from this ranking.

### volatility_target

- Highest reported Sharpe in this diagnostic family: Maximum Sharpe Vol Target 8% at `0.08` (0.58).
- Lowest reported Sharpe in this diagnostic family: Equal Weight Vol Target 8% at `0.08` (0.26).
- This comparison is descriptive; the best value was not selected for confirmation from this ranking.

### trend_moving_average_days

- Highest reported Sharpe in this diagnostic family: Trend Filtered Minimum Variance at `150` (0.77).
- Lowest reported Sharpe in this diagnostic family: Trend Filtered Equal Weight at `250` (0.38).
- This comparison is descriptive; the best value was not selected for confirmation from this ranking.

### momentum_lookback_days

- Highest reported Sharpe in this diagnostic family: Dual Momentum Equal Weight at `252` (0.60).
- Lowest reported Sharpe in this diagnostic family: Dual Momentum Equal Weight at `189` (0.49).
- This comparison is descriptive; the best value was not selected for confirmation from this ranking.

### extra_execution_delay_days

- Highest reported Sharpe in this diagnostic family: Trend Filtered Minimum Variance at `0` (0.64).
- Lowest reported Sharpe in this diagnostic family: Trend Filtered Equal Weight at `1` (0.32).
- This comparison is descriptive; the best value was not selected for confirmation from this ranking.

### expected_return_estimator

- Highest reported Sharpe in this diagnostic family: Maximum Sharpe at `grand_mean_shrink` (0.60).
- Lowest reported Sharpe in this diagnostic family: Maximum Sharpe at `ewma` (0.47).
- This comparison is descriptive; the best value was not selected for confirmation from this ranking.

The complete experiment matrix is available in `outputs/parameter_sensitivity.csv`.