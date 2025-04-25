#!/usr/bin/env python3
# trading_basic_and_topk_portfolio.py
#
# Two strategies, same horizon & capital:
#   • Basic-sign  : long every pred>0, short every pred<0 (equal-weight)
#   • Top-k       : long top-k, short bottom-k (equal-weight)
# All trades at the same *index position* happen in the same “period”.
# ---------------------------------------------------------------------

import json, math, pathlib, sys
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 1 . User parameters
# ------------------------------------------------------------
RESULTS_FILE = "arima_lstm_vwap"   # your *.jsonl*
jsonl_path   = pathlib.Path("results") / pathlib.Path(f"{RESULTS_FILE}.jsonl")
plot_path    = pathlib.Path("plots") / pathlib.Path(f"{RESULTS_FILE}")
init_capital = 1_000_000.0                        # starting NAV ($)
topk         = 5                                  # k longs & k shorts
ann_factor   = math.sqrt(252)                     # Sharpe scaling
ARIMA = False

# ------------------------------------------------------------
# 2 . Load data
# ------------------------------------------------------------
stocks_preds, stocks_trues = [], []
with jsonl_path.open() as fh:
    for line in fh:
        rec   = json.loads(line)
        if ARIMA:
            RESULTS_FILE = "base_arima"
            preds = np.asarray(rec["arima"], dtype=float)
        else:
            preds = np.asarray(rec["hybrid"], dtype=float)
        trues = np.asarray(rec["y_test"], dtype=float)
        if len(preds) == len(trues):
            stocks_preds.append(preds)
            stocks_trues.append(trues)

n_stocks = len(stocks_preds)
if n_stocks == 0:
    sys.exit("No usable stocks found.")
print(f"Loaded {n_stocks} stocks.")

max_len = max(map(len, stocks_preds))             # #periods

# ------------------------------------------------------------
# 3 . Helpers
# ------------------------------------------------------------
def annualised_sharpe(r):
    mu, sig = r.mean(), r.std(ddof=1)
    return 0.0 if sig < 1e-12 else (mu/sig)*ann_factor

def plot_curve(curve, title, fname):
    plt.figure(figsize=(10,4))
    plt.plot(curve, lw=1)
    plt.title(title)
    plt.xlabel("Period")
    plt.ylabel("Equity ($)")
    plt.grid(alpha=.3); plt.tight_layout()
    plt.savefig(fname, dpi=300); plt.close()

# ------------------------------------------------------------
# 4 . BASIC-SIGN portfolio (equal-weight every ± signal)
# ------------------------------------------------------------
basic_nav     = init_capital
basic_curve   = []
basic_rets    = []

for idx in range(max_len):
    preds_i, trues_i = [], []
    for p,t in zip(stocks_preds, stocks_trues):
        if idx < len(p):
            preds_i.append(p[idx])
            trues_i.append(t[idx])
    if not preds_i:                                  # no stocks left
        continue

    preds_i  = np.array(preds_i)
    trues_i  = np.array(trues_i)

    long_mask  = preds_i > 0
    short_mask = preds_i < 0

    n_long  = long_mask.sum()
    n_short = short_mask.sum()
    if n_long + n_short == 0:
        basic_curve.append(basic_nav); basic_rets.append(0); continue

    w = np.zeros_like(preds_i, dtype=float)
    if n_long:   w[long_mask]  =  1.0 / n_long
    if n_short:  w[short_mask] = -1.0 / n_short       # dollar-neutral

    pct = (w * trues_i).sum()
    basic_nav *= (1 + pct)
    basic_curve.append(basic_nav)
    basic_rets.append(pct)

basic_curve = np.array(basic_curve)
basic_rets  = np.array(basic_rets)

# ------------------------------------------------------------
# 5 . TOP-k / BOTTOM-k portfolio
# ------------------------------------------------------------
topk_nav     = init_capital
topk_curve   = []
topk_rets    = []

for idx in range(max_len):
    preds_i, trues_i = [], []
    for p,t in zip(stocks_preds, stocks_trues):
        if idx < len(p):
            preds_i.append(p[idx])
            trues_i.append(t[idx])
    if len(preds_i) < 2*topk:
        continue

    preds_i  = np.array(preds_i)
    trues_i  = np.array(trues_i)
    order    = preds_i.argsort()                      # ascending

    w = np.zeros_like(preds_i, dtype=float)
    w[order[-topk:]] =  0.5 / topk
    w[order[:topk]]  = -0.5 / topk

    pct = (w * trues_i).sum()
    topk_nav *= (1 + pct)
    topk_curve.append(topk_nav)
    topk_rets.append(pct)

topk_curve = np.array(topk_curve)
topk_rets  = np.array(topk_rets)

# ------------------------------------------------------------
# 6 . Results
# ------------------------------------------------------------
print("\n=== BASIC-SIGN ===")
print(f"Final NAV   : ${basic_curve[-1]:,.2f}")
print(f"Total return: {(basic_curve[-1]/init_capital-1)*100:.2f} %")
print(f"Sharpe      : {annualised_sharpe(basic_rets):.2f}")

print("\n=== TOP-{0}/BOTTOM-{0} ===".format(topk))
print(f"Final NAV   : ${topk_curve[-1]:,.2f}")
print(f"Total return: {(topk_curve[-1]/init_capital-1)*100:.2f} %")
print(f"Sharpe      : {annualised_sharpe(topk_rets):.2f}")

plot_curve(basic_curve,
           f"Basic-sign | Sharpe {annualised_sharpe(basic_rets):.2f}",
           plot_path.parent / f"{plot_path.stem}_basic_equity.png")
plot_curve(topk_curve,
           f"Top-{topk} | Sharpe {annualised_sharpe(topk_rets):.2f}",
           plot_path.parent / f"{plot_path.stem}_top{topk}_equity.png")

print("\nPNG equity curves saved beside the JSONL file.")
