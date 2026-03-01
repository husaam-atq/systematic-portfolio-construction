from __future__ import annotations
import argparse
import json
import os
import pandas as pd

from src.data import download_prices, returns
from src.factors import momentum_12_1, low_vol, reversal_1w, to_signal
from src.portfolio import combine_signals, long_short_weights
from src.risk import vol_target_scale
from src.backtest import backtest
from src.metrics import summary
from src.plots import equity_curve

DEFAULT_TICKERS = [
    "AAPL","MSFT","AMZN","GOOGL","META","NVDA","TSLA","JPM","JNJ","PG",
    "XOM","CVX","KO","PEP","WMT","HD","UNH","MRK","ABBV","COST",
    "V","MA","CRM","ADBE","NFLX","DIS","INTC","CSCO","ORCL","BAC"
]

def run(start: str, tc_bps: float, outdir: str, target_vol: float):
    os.makedirs(outdir, exist_ok=True)

    px = download_prices(DEFAULT_TICKERS, start=start)
    rets = returns(px).dropna(how="all")

    # factors -> signals
    sig_mom = to_signal(momentum_12_1(px))
    sig_low = to_signal(low_vol(px))
    sig_rev = to_signal(reversal_1w(px))

    # align to returns dates
    sig_mom = sig_mom.reindex(rets.index)
    sig_low = sig_low.reindex(rets.index)
    sig_rev = sig_rev.reindex(rets.index)

    # weights for each factor alone
    w_mom = long_short_weights(sig_mom)
    w_low = long_short_weights(sig_low)
    w_rev = long_short_weights(sig_rev)

    # multi-factor blend (equal weights)
    combo_sig = combine_signals(
        {"mom": sig_mom, "lowvol": sig_low, "reversal": sig_rev},
        {"mom": 1/3, "lowvol": 1/3, "reversal": 1/3}
    )
    w_combo = long_short_weights(combo_sig)

    # vol targeting on multi-factor portfolio
    w_combo_scaled = vol_target_scale(w_combo, rets, target_ann_vol=target_vol)

    # backtests
    bt_mom = backtest(rets, w_mom, tc_bps=tc_bps)
    bt_low = backtest(rets, w_low, tc_bps=tc_bps)
    bt_rev = backtest(rets, w_rev, tc_bps=tc_bps)
    bt_combo = backtest(rets, w_combo_scaled, tc_bps=tc_bps)

    stats = pd.DataFrame({
        "Momentum": summary(bt_mom["net"]),
        "LowVol": summary(bt_low["net"]),
        "Reversal": summary(bt_rev["net"]),
        "MultiFactor(VolTarget)": summary(bt_combo["net"]),
    })

    print("\n=== Summary (NET) ===")
    print(stats.T.to_string(float_format=lambda x: f"{x:0.4f}"))

    # save
    stats_path = os.path.join(outdir, "stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats.T.to_dict(orient="index"), f, indent=2)

    bt_combo.to_csv(os.path.join(outdir, "backtest_multifactor.csv"))

    equity_curve(bt_combo["net"], f"Equity Curve (Net) - MultiFactor VolTarget={target_vol:.0%}",
                 path=os.path.join(outdir, "equity_multifactor.png"))

    print(f"\nSaved: {stats_path}")
    print(f"Saved: {os.path.join(outdir, 'backtest_multifactor.csv')}")
    print(f"Saved: {os.path.join(outdir, 'equity_multifactor.png')}")

def main():
    p = argparse.ArgumentParser(description="Systematic Portfolio Construction (Multi-Factor)")
    p.add_argument("--start", default="2018-01-01")
    p.add_argument("--tc_bps", type=float, default=10.0)
    p.add_argument("--outdir", default="outputs")
    p.add_argument("--target_vol", type=float, default=0.12)
    args = p.parse_args()
    run(args.start, args.tc_bps, args.outdir, args.target_vol)

if __name__ == "__main__":
    main()
