# Baseline Reproduction Report

## Provenance

The existing engine was reproduced before methodology changes on 2026-08-25 from commit `48bcbe9eb2bba34104e15365bab98d4277585a1a`. The run requested adjusted ETF prices only through 2024-12-31. No 2025 or later portfolio result was requested or inspected.

The complete generated CSV and PNG set is preserved under `archive/baseline_48bcbe9/`, together with the exact historical configuration.

## Historical Design

- Universe: SPY, EFA, EEM, TLT, IEF, SHY, GLD, VNQ and DBC.
- Development sample: 2010-01-01 through 2024-12-31.
- Estimation window: 252 trading days.
- Rebalancing: monthly.
- Constraints: long-only, 40% maximum asset weight for optimised methods.
- Costs: 5 bps multiplied by the sum of absolute applied-weight changes.
- Volatility targeting: 63-day realised volatility, 8%/10%/12% targets, 1.5x cap.
- Cash and financing: uninvested cash earned zero and leverage incurred no financing cost.
- Performance metrics: zero risk-free rate.

## Reproduced Headline Results

| Method | CAGR | Sharpe | Max drawdown |
|---|---:|---:|---:|
| Equal Weight | 4.86% | 0.57 | -18.21% |
| Minimum Variance | 2.84% | 0.78 | -12.91% |
| Dual Momentum Equal Weight | 8.31% | 0.75 | -18.17% |
| Maximum Sharpe | 4.36% | 0.64 | -15.89% |
| Maximum Sharpe Vol Target 10% | 6.11% | 0.75 | -18.93% |

The Maximum Sharpe family varied modestly across repeated downloads/runs while the other headline methods were stable. That is consistent with the method's sensitivity to noisy sample-mean expected returns and is itself baseline evidence.

## Baseline Weaknesses Preserved

1. Applied target weights were forward-filled between rebalances, so holdings did not drift with realised asset returns.
2. Turnover compared abstract applied weights, not the trades required to rebalance drifted holdings.
3. Volatility targeting treated exposure below 1x as zero-return cash and exposure above 1x as costless financing.
4. Changing leverage created turnover, but leverage costs were absent.
5. Risk contribution reporting used sample covariance for every strategy, including covariance-shrinkage variants.
6. The engine did not include a traditional balanced benchmark.
7. Full-sample rankings dominated the presentation; no formal walk-forward segment protocol or dependence-aware uncertainty intervals were reported.
8. Optimiser convergence, condition numbers, bound frequency and perturbation sensitivity were not recorded.
9. Missing observations were forward-filled before complete-case filtering without a formal data-quality report.
10. No automated test suite or continuous-integration workflow protected chronology and accounting assumptions.

These weaknesses motivate the upgraded research design. Historical baseline results are not targets for the new implementation.
