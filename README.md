# Systematic Portfolio Construction (Multi-Factor)

A practical portfolio construction project demonstrating how cross-sectional factor signals can be blended into a systematic long/short portfolio.

This repo focuses on:
- Multi-factor signal blending (momentum, low-volatility, short-term reversal)
- Long/short portfolio formation (quantile-based)
- Volatility targeting (risk scaling)
- Transaction cost + turnover modelling
- Performance evaluation and equity curve generation

## Run

Install:
pip install -r requirements.txt

Run:
python -m src.cli --start 2018-01-01 --tc_bps 10 --target_vol 0.12

Outputs are saved in outputs/.

## Example Output

![Equity Curve](outputs/equity_multifactor.png)


## Notes

Results show raw factor behaviour and a simple multi-factor blend with risk targeting. The framework is designed to be extended with:
- Beta/sector neutralisation
- Monthly rebalancing
- Multi-factor optimisation
- Risk decomposition
