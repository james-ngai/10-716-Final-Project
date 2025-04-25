#!/usr/bin/env python3
# trading_compare_models.py
#
# Compare multiple models' trading performance:
#   • Basic-sign  : long every pred>0, short every pred<0 (equal-weight)
#   • Top-k       : long top-k, short bottom-k (equal-weight)
# Overlay their equity curves on a single plot.
# ---------------------------------------------------------------------

import json, math, pathlib, sys
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 1 . User parameters
# ------------------------------------------------------------
RESULTS_FILES = [
    "arima_lstm_vwap",
    "arima_lstm_novwap",
    "arima_transformer_vwap",
    "arima_transformer_novwap",
    "arima_baseline",
]  # list of *.jsonl* stem names
jsonl_dir = pathlib.Path("results")
plot_dir  = pathlib.Path("plots")
init_capital = 1_000_000.0
topk         = 5
ann_factor   = math.sqrt(252)
ARIMA_MODELS = {"arima_baseline"}  # if filename in this set, treat it as ARIMA

# ------------------------------------------------------------
# 2 . Helpers
# ------------------------------------------------------------
def annualised_sharpe(r):
    mu, sig = r.mean(), r.std(ddof=1)
    return 0.0 if sig < 1e-12 else (mu/sig)*ann_factor

def load_preds_trues(jsonl_path, is_arima):
    stocks_preds, stocks_trues = [], []
    with jsonl_path.open() as fh:
        for line in fh:
            rec = json.loads(line)
            preds = np.asarray(rec["arima"] if is_arima else rec["hybrid"], dtype=float)
            trues = np.asarray(rec["y_test"], dtype=float)
            if len(preds) == len(trues):
                stocks_preds.append(preds)
                stocks_trues.append(trues)
    return stocks_preds, stocks_trues

def simulate_portfolio(stocks_preds, stocks_trues, topk, init_capital):
    max_len = max(map(len, stocks_preds))
    basic_nav, topk_nav = init_capital, init_capital
    basic_curve, basic_rets = [], []
    topk_curve, topk_rets = [], []

    for idx in range(max_len):
        preds_i, trues_i = [], []
        for p, t in zip(stocks_preds, stocks_trues):
            if idx < len(p):
                preds_i.append(p[idx])
                trues_i.append(t[idx])
        if not preds_i:
            continue

        preds_i = np.array(preds_i)
        trues_i = np.array(trues_i)

        # BASIC-SIGN
        long_mask  = preds_i > 0
        short_mask = preds_i < 0
        n_long  = long_mask.sum()
        n_short = short_mask.sum()

        if n_long + n_short == 0:
            basic_curve.append(basic_nav)
            basic_rets.append(0)
        else:
            w = np.zeros_like(preds_i, dtype=float)
            if n_long:  w[long_mask]  =  1.0 / n_long
            if n_short: w[short_mask] = -1.0 / n_short
            pct = (w * trues_i).sum()
            basic_nav *= (1 + pct)
            basic_curve.append(basic_nav)
            basic_rets.append(pct)

        # TOP-k
        if len(preds_i) >= 2*topk:
            order = preds_i.argsort()  # ascending
            w = np.zeros_like(preds_i, dtype=float)
            w[order[-topk:]] =  0.5 / topk
            w[order[:topk]]  = -0.5 / topk
            pct = (w * trues_i).sum()
            topk_nav *= (1 + pct)
            topk_curve.append(topk_nav)
            topk_rets.append(pct)

    return np.array(basic_curve), np.array(basic_rets), np.array(topk_curve), np.array(topk_rets)

# ------------------------------------------------------------
# 3 . Main loop
# ------------------------------------------------------------
all_curves = {}

for fname in RESULTS_FILES:
    print(f"\n=== Processing {fname} ===")
    jsonl_path = jsonl_dir / f"{fname}.jsonl"
    is_arima   = fname in ARIMA_MODELS

    stocks_preds, stocks_trues = load_preds_trues(jsonl_path, is_arima)
    if not stocks_preds:
        print(f"  [!] No usable stocks found for {fname}. Skipping.")
        continue

    basic_curve, basic_rets, topk_curve, topk_rets = simulate_portfolio(
        stocks_preds, stocks_trues, topk, init_capital
    )

    # Save curves
    all_curves[f"{fname}_basic"] = basic_curve
    all_curves[f"{fname}_topk"]  = topk_curve

    # Print results
    print("--- BASIC-SIGN ---")
    print(f"Final NAV   : ${basic_curve[-1]:,.2f}")
    print(f"Total return: {(basic_curve[-1]/init_capital-1)*100:.2f} %")
    print(f"Sharpe      : {annualised_sharpe(basic_rets):.2f}")

    print(f"--- TOP-{topk}/BOTTOM-{topk} ---")
    print(f"Final NAV   : ${topk_curve[-1]:,.2f}")
    print(f"Total return: {(topk_curve[-1]/init_capital-1)*100:.2f} %")
    print(f"Sharpe      : {annualised_sharpe(topk_rets):.2f}")

# ------------------------------------------------------------
# 4 . Plot overlay
# ------------------------------------------------------------
plt.figure(figsize=(12,6))
for label, curve in all_curves.items():
    plt.plot(curve, lw=1, label=label)

plt.title(f"Equity Curves | Basic-sign and Top-{topk}")
plt.xlabel("Period")
plt.ylabel("Equity ($)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plot_dir.mkdir(exist_ok=True)
out_fname = plot_dir / "compare_models_equity.png"
plt.savefig(out_fname, dpi=300)
plt.close()

print(f"\nOverlay PNG saved to {out_fname}")
