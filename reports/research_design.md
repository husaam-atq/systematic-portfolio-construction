# Research Design and Methodology Audit

## Research Question

Which portfolio-construction methods remain useful once estimation error, concentration, costs and changing market regimes are taken seriously?

The goal is robustness, not maximisation of historical Sharpe or CAGR. Static risk allocation and tactical overlays will be evaluated separately.

## Chronology and Holdout Discipline

- Development research is fixed to 2010-01-01 through 2024-12-31.
- No 2025+ portfolio result may be requested, inspected or calculated until the methodology is implemented, tested, reviewed, frozen, hashed and committed.
- The frozen confirmation interval will begin on 2025-01-01 and end at the latest completed trading date specified in the frozen artifact.
- After confirmation data are viewed, methodology, representative methods and evaluation metrics may not be changed in response.

## Audit Findings

### Timing and Simulation

- Existing monthly decisions are shifted before return application, and rolling estimators exclude the decision-date return. Direct same-day leakage was not found.
- Existing simulation forward-fills target weights, preventing holdings drift.
- Existing turnover is the sum of absolute changes in applied targets, not trades from drifted pre-trade holdings.
- Trend and dual-momentum signals are price-lagged, and volatility scaling is shifted. These controls are causal but will be simplified around a single decision-at-close / earn-next-return convention.

### Cash, Leverage and Costs

- Exposure below 1x currently earns zero rather than a public cash return.
- Exposure above 1x currently has no financing charge.
- The upgraded simulator will use SHY as a reproducible cash proxy. Positive cash earns the SHY total return. Negative cash pays the SHY return plus a fixed 50 bps annual financing spread.
- Volatility-target exposure remains capped at 1.5x. All leverage changes and rebalance trades enter turnover at 5 bps one-way by default.

### Estimation and Optimisation

- Sample covariance is regularised only by a tiny diagonal jitter unless explicit diagonal shrinkage is selected.
- Maximum Sharpe uses annualised sample means, a particularly noisy expected-return estimate.
- Optimiser failure silently falls back to initial weights and convergence diagnostics are not retained.
- The upgraded engine will retain the methods but record solver success, condition number, bound frequency, concentration and perturbation sensitivity.
- HRP will be added as one bounded-complexity robustness method because it avoids expected-return estimation and direct covariance inversion.

### Risk and Benchmarking

- Risk contributions are mathematically component contributions to variance, but shrinkage methods are currently evaluated with sample covariance rather than their own estimator.
- The upgraded engine will retain the covariance used by each decision and numerically verify equal-risk-contribution error for Risk Parity.
- Equal Weight remains the primary implementation benchmark.
- A fixed 60% SPY / 40% IEF portfolio will be added as a transparent traditional balanced benchmark using the same simulation and cost framework. SPY remains informational only.

## Frozen Development Defaults

Defaults are chosen before upgraded development results are inspected:

- Universe: SPY, EFA, EEM, TLT, IEF, SHY, GLD, VNQ and DBC.
- Base estimation window: 252 trading days.
- Base rebalance frequency: monthly.
- Default optimised-method cap: 40% per ETF.
- Default diagonal covariance shrinkage: 25%.
- Transaction cost: 5 bps per one-way ETF trade.
- Positive cash return: SHY total return.
- Financing cost: SHY total return plus 50 bps annual spread for negative cash.
- Volatility target robustness levels: 8%, 10% and 12%; 63-day trailing estimator; 1.5x cap.
- Trend rule: close above 200-day moving average, assessed at the decision close and applied to the next return.
- Dual momentum: 12-month return excluding the most recent 21 trading days; top three positive risky ETFs; residual to SHY.
- Statistical uncertainty: moving-block bootstrap with 21-day blocks, 500 deterministic resamples and 95% intervals.

## Canonical Methods

Static allocation:

- Equal Weight
- Traditional 60/40 (SPY/IEF)
- Inverse Volatility
- Minimum Variance
- Minimum Variance with 25% diagonal shrinkage
- Maximum Sharpe
- Maximum Sharpe with 25% diagonal shrinkage
- Risk Parity / Equal Risk Contribution
- Risk Parity with 25% diagonal shrinkage
- Hierarchical Risk Parity

Risk-management overlays:

- 10% volatility-target variants of Equal Weight, Minimum Variance, Maximum Sharpe and Risk Parity in the main comparison.
- 8% and 12% targets retained as sensitivity evidence.

Tactical methods, reported separately:

- Trend Filtered Equal Weight
- Trend Filtered Minimum Variance
- Trend Filtered Risk Parity
- Dual Momentum Equal Weight
- Dual Momentum Inverse Volatility

## Development Evaluation

Rolling monthly decisions are genuine walk-forward estimates. Segment evidence will be reported for fixed calendar OOS segments after the initial warm-up, with return, Sharpe, volatility, drawdown, turnover, concentration, risk concentration and benchmark-relative return.

Parameter sensitivity is diagnostic rather than an optimisation grid:

- estimation windows: 126, 252 and 504 days;
- monthly and quarterly rebalancing;
- covariance shrinkage: 0%, 25%, 50% and 75%;
- asset caps: 30%, 40% and 60%;
- transaction costs: 0, 5, 10 and 20 bps;
- volatility targets: 8%, 10% and 12%;
- tactical lookbacks: fixed reasonable alternatives reported without selecting the best full-sample result.

No parameter will be selected from the final confirmation period.

## Predefined Stress Episodes

| Episode | Start | End |
|---|---|---|
| Euro-area / sovereign stress | 2011-07-01 | 2011-10-04 |
| 2013 rate shock | 2013-05-02 | 2013-09-05 |
| Growth / commodity stress | 2015-07-01 | 2016-02-11 |
| Q4 2018 equity sell-off | 2018-10-01 | 2018-12-24 |
| COVID shock | 2020-02-19 | 2020-03-23 |
| Inflation / rate shock | 2022-01-03 | 2022-10-14 |
| Recovery / changed rate regime | 2023-01-03 | 2024-12-31 |

## Development Decision Rule

No single score determines the representative portfolio. Conclusions will weigh OOS stability, drawdown, concentration, risk contribution, turnover, cost resilience, parameter sensitivity, estimator stability, simplicity and benchmark-relative performance. If performance differences are statistically indistinguishable, the simpler method receives preference.
