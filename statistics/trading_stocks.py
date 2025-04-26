#!/usr/bin/env python3
# trading_compare_models_longonly.py
#
# Compare multiple models' trading performance with LONG-ONLY strategies:
#   • Basic-long  : long every pred>0  (equal-weight, no shorts)
#   • Top-k-long  : long top-k         (equal-weight, no shorts)
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
]                                   # *.jsonl* stem names
jsonl_dir    = pathlib.Path("results")
plot_dir     = pathlib.Path("plots")
init_capital = 1_000_000.0          # starting NAV ($)
topk         = 10                   # LONG ONLY: buy top-k predictions
ann_factor   = math.sqrt(252)       # Sharpe scaling
ARIMA_MODELS = {"arima_baseline"}   # filenames treated as ARIMA (no “hybrid” key)
DAILY_T_BILL_INSTEREST = 0.0000795

# ------------------------------------------------------------
# 2 . Helpers
# ------------------------------------------------------------
def annualised_sharpe(r):
    non_zero = r[r != 0.0]
    if non_zero.size <= 1:          # not enough data for a stdev
        return 0.0
    mu, sig = r.mean(), non_zero.std(ddof=1)
    return 0.0 if sig < 1e-12 else ((mu - DAILY_T_BILL_INSTEREST)/ sig) * ann_factor

def load_preds_trues(jsonl_path, is_arima):
    """Return two lists (pred arrays, true-return arrays) – one per stock."""
    preds_list, trues_list = [], []
    with jsonl_path.open() as fh:
        for line in fh:
            rec   = json.loads(line)
            preds = np.asarray(rec["arima"] if is_arima else rec["hybrid"], dtype=float)
            trues = np.asarray(rec["y_test"], dtype=float)
            if len(preds) == len(trues):          # discard failed fits, etc.
                preds_list.append(preds)
                trues_list.append(trues)
    return preds_list, trues_list

def simulate_portfolio(stocks_preds, stocks_trues, topk, init_capital):
    """
    Returns four 1-D arrays:
      basic_curve, basic_rets, topk_curve, topk_rets
    All PnL comes from *appreciation* of long positions – never short.
    """
    max_len       = max(map(len, stocks_preds))
    basic_NAV     = init_capital
    topk_NAV      = init_capital
    basic_curve   = []
    topk_curve    = []
    basic_rets    = []
    topk_rets     = []

    for t in range(max_len):
        # Gather t-th prediction/true for every stock that still has data
        preds_t, trues_t = [], []
        for p, q in zip(stocks_preds, stocks_trues):
            if t < len(p):
                preds_t.append(p[t])
                trues_t.append(q[t])
        if not preds_t:                          # no live stocks this step
            continue

        preds_t = np.asarray(preds_t)
        trues_t = np.asarray(trues_t)

        # ---------------- Basic-long: buy everything with positive signal ----
        long_mask = preds_t > 0
        n_long    = long_mask.sum()

        if n_long == 0:                          # sit in cash
            basic_curve.append(basic_NAV)
            basic_rets.append(0.0)
        else:
            w           = np.zeros_like(preds_t, dtype=float)
            w[long_mask] = 1.0 / n_long          # equal-weight, sums to 1
            pct_ret     = (w * trues_t).sum()    # portfolio % move this step
            basic_NAV  *= (1.0 + pct_ret)
            basic_curve.append(basic_NAV)
            basic_rets.append(pct_ret)

        # ---------------- Top-k-long: buy k best signals, equal-weight -------
        k = min(topk, len(preds_t))              # guard against <k stocks
        order       = preds_t.argsort()          # ascending
        winners_idx = order[-k:]                 # highest k predictions
        w           = np.zeros_like(preds_t, dtype=float)
        w[winners_idx] = 1.0 / k
        pct_ret     = (w * trues_t).sum()
        topk_NAV   *= (1.0 + pct_ret)
        topk_curve.append(topk_NAV)
        topk_rets.append(pct_ret)

    return (np.asarray(basic_curve), np.asarray(basic_rets),
            np.asarray(topk_curve),  np.asarray(topk_rets))

# ------------------------------------------------------------
# 3 . Main loop
# ------------------------------------------------------------
all_curves = {}

for fname in RESULTS_FILES:
    print(f"\n=== Processing {fname} ===")
    jsonl_path = jsonl_dir / f"{fname}.jsonl"
    is_arima   = fname in ARIMA_MODELS

    preds, trues = load_preds_trues(jsonl_path, is_arima)
    if not preds:
        print("  [!] No usable stocks – skipping.")
        continue

    basic_curve, basic_rets, topk_curve, topk_rets = simulate_portfolio(
        preds, trues, topk, init_capital
    )

    # cache for overlay
    all_curves[f"{fname}_basic_long"] = basic_curve
    all_curves[f"{fname}_top{topk}_long"] = topk_curve

    # ---- reporting ----
    print("--- BASIC-LONG (all pred>0) ---")
    print(f"Final NAV   : ${basic_curve[-1]:,.2f}")
    print(f"Total return: {(basic_curve[-1]/init_capital - 1)*100:.2f} %")
    print(f"Sharpe      : {annualised_sharpe(basic_rets):.2f}")

    print(f"--- TOP-{topk}-LONG ---")
    print(f"Final NAV   : ${topk_curve[-1]:,.2f}")
    print(f"Total return: {(topk_curve[-1]/init_capital - 1)*100:.2f} %")
    print(f"Sharpe      : {annualised_sharpe(topk_rets):.2f}")

# ------------------------------------------------------------
# 4 . Plot overlay
# ------------------------------------------------------------
plt.figure(figsize=(12, 6))
for label, curve in all_curves.items():
    plt.plot(curve, lw=1, label=label)

plt.title(f"Equity Curves | Long-only: Basic & Top-{topk}")
plt.xlabel("Period")
plt.ylabel("Equity ($)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plot_dir.mkdir(exist_ok=True)
out_png = plot_dir / f"compare_models_equity_longonly_top{topk}.png"
plt.savefig(out_png, dpi=300)
plt.close()

print(f"\nOverlay PNG saved to {out_png}")
