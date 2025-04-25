#!/usr/bin/env python3
# arima_lstm_residual_smallstocks.py
#
# Train the ARIMA-residual-plus-LSTM hybrid **for every stock that has
# fewer than MAX_POINTS rows** and dump one JSON object per stock to a
# single JSONL file.
#
# USAGE ---------------------------------------------------------------
"""
python3 final_tests/arima_lstm__vwap_residual.py \
    --data ./data/dict_of_data_Jan2025_part1.npy \
    --out results/arima_lstm__vwap_residual.jsonl

"""
# --------------------------------------------------------------------

import argparse, json, warnings, sys
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------------------ #
# 1.  CLI                                                            #
# ------------------------------------------------------------------ #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Batch ARIMA+LSTM residual hybrid (small-stock mode)")
    p.add_argument("--data",   required=True, type=str)
    p.add_argument("--data2",  default=None,  type=str)
    # selection
    p.add_argument("--max-points", type=int, default=500,
                   help="Train only on stocks with > THIS many rows")
    # model hyper-params (unchanged defaults)
    p.add_argument("--window", type=int, default=10)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch",  type=int, default=256)
    p.add_argument("--lr",     type=float, default=1e-3)
    p.add_argument("--split",  type=float, default=0.8)
    # output handling
    p.add_argument("--out",    type=str, default="gg.jsonl",
                   help="Path to output JSONL file")
    p.add_argument("--append", action="store_true",
                   help="Append to existing file (otherwise overwrite)")
    return p.parse_args()


# ------------------------------------------------------------------ #
# 2.  Lightweight dataset + model helpers (unchanged)                #
# ------------------------------------------------------------------ #
class ResidualDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, seq: int):
        assert len(X) == len(y)
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))
        self.seq = seq

    def __len__(self):          return max(0, len(self.X) - self.seq)
    def __getitem__(self, idx): return (self.X[idx:idx+self.seq],
                                        self.y[idx + self.seq])


class LSTMResidual(nn.Module):
    def __init__(self, input_dim: int, hidden: int, layers: int):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, layers, batch_first=True)
        self.fc   = nn.Linear(hidden, 1)

    def forward(self, x):
        h, _ = self.lstm(x)
        return self.fc(h[:, -1]).squeeze(-1)


def rolling_arima(series: np.ndarray, order=(1, 1, 1)) -> np.ndarray:
    p, d, q = order
    min_hist = max(p, d, q) + 1
    preds = np.full_like(series, np.nan, dtype=np.float32)
    history = list(series[:min_hist])
    for t in range(min_hist, len(series)):
        try:
            preds[t] = float(ARIMA(history, order=order).fit().forecast()[0])
        except Exception:  # fallback: last value
            preds[t] = history[-1]
        history.append(series[t])
    return preds


def compute_vwap_feats(df: pd.DataFrame) -> pd.DataFrame:
    df["vwap"] = (df["high"] + df["low"] + df["close"]) / 3.0
    df = df.sort_values(["stock_id", "day_idx"])
    df["vwap_5"]  = df.groupby("stock_id")["vwap"].transform(lambda x: x.rolling(5, 1).mean())
    df["dev_vwap"] = (df["close"] - df["vwap"]) / df["vwap"]
    df["trend_flag"] = (df["close"] > df["vwap"]).astype(np.float32)
    return df


# ------------------------------------------------------------------ #
# 3.  Per-stock training routine                                     #
# ------------------------------------------------------------------ #
def train_one_stock(stock_id: int,
                    idx_mask: np.ndarray,
                    data: dict,
                    df: pd.DataFrame,
                    args: argparse.Namespace) -> dict | None:
    """
    Returns the result dictionary or None if the stock is skipped
    (e.g. not enough data after window/ARIMA trimming).
    """
    # raw feature matrix (200 alpha signals) & label
    X_raw = data["x_data"][idx_mask]         # (T, 200)
    y_all = data["y_data"][idx_mask]         # (T,)

    # Add VWAP-derived cols (3)  →  (T, 203)
    vwap_cols = df.loc[idx_mask, ["vwap_5", "dev_vwap", "trend_flag"]].to_numpy(dtype=np.float32)
    X_with_vwap = np.concatenate([X_raw, vwap_cols], axis=1)

    # ARIMA one-step forecasts
    arima_full = rolling_arima(y_all, order=(1, 1, 1))
    valid = ~np.isnan(arima_full)
    if valid.sum() < args.window + 5:        # not enough usable samples
        print(f"[{stock_id:>5}]  skipped – too short after ARIMA ({valid.sum()} rows)")
        return None

    X_with_vwap, y_all, arima_full = X_with_vwap[valid], y_all[valid], arima_full[valid]
    X_aug = np.concatenate([X_with_vwap, arima_full.reshape(-1, 1)], axis=1)  # (T', 204)

    # Train / test split
    split = int(len(y_all) * args.split)
    X_tr, X_te = X_aug[:split], X_aug[split:]
    y_tr, y_te = y_all[:split], y_all[split:]
    arima_tr, arima_te = arima_full[:split], arima_full[split:]

    resid_tr, resid_te = y_tr - arima_tr, y_te - arima_te
    seq = args.window

    # Dataloaders
    ds_tr = ResidualDataset(X_tr, resid_tr, seq)
    ds_te = ResidualDataset(X_te, resid_te, seq)
    if len(ds_tr) == 0 or len(ds_te) == 0:
        print(f"[{stock_id:>5}]  skipped – no windowed batches after split")
        return None

    dl_tr = DataLoader(ds_tr, batch_size=args.batch, shuffle=True)
    dl_te = DataLoader(ds_te, batch_size=args.batch)

    # LSTM
    net = LSTMResidual(input_dim=X_aug.shape[1],
                       hidden=args.hidden,
                       layers=args.layers).to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    loss_fn = torch.nn.MSELoss()

    for _ in range(args.epochs):
        net.train()
        for xb, yb in dl_tr:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(); loss_fn(net(xb), yb).backward(); opt.step()

    # Test prediction
    net.eval(); resid_pred = []
    with torch.no_grad():
        for xb, _ in dl_te:
            resid_pred.append(net(xb.to(DEVICE)).cpu())
    resid_pred = torch.cat(resid_pred).numpy()

    skip = seq               # because first `seq` labels are lost in dataset slicing
    y_te_cut   = y_te[skip:]
    arima_cut  = arima_te[skip:]
    resid_true = resid_te[skip:]
    hybrid     = arima_cut + resid_pred

    result = {
        "stock_id":   int(stock_id),
        "n_points":   int(idx_mask.sum()),
        "mse_arima":  float(mean_squared_error(y_te_cut, arima_cut)),
        "mae_arima":  float(mean_absolute_error(y_te_cut, arima_cut)),
        "mse_hybrid": float(mean_squared_error(y_te_cut, hybrid)),
        "mae_hybrid": float(mean_absolute_error(y_te_cut, hybrid)),
        "r2_hybrid":  float(r2_score(y_te_cut, hybrid)),
        "hyperparams": {**vars(args)},
        # artefacts for optional deep-dive / plotting  (can be huge → keep only if needed)
        "y_test":     y_te_cut.tolist(),
        "arima":      arima_cut.tolist(),
        "resid_true": resid_true.tolist(),
        "resid_pred": resid_pred.tolist(),
        "hybrid":     hybrid.tolist(),
        "vwap_cols":  ["vwap_5", "dev_vwap", "trend_flag"],
    }
    return result


# ------------------------------------------------------------------ #
# 4.  Main driver                                                    #
# ------------------------------------------------------------------ #
def main() -> None:
    args = parse_args()
    ##############################################################
    # 4.1  Load & merge part-files                               #
    ##############################################################
    d1 = np.load(args.data, allow_pickle=True).item()
    data = d1 if args.data2 is None else {
        k: np.concatenate([d1[k], np.load(args.data2, allow_pickle=True).item()[k]])
        for k in d1
    }

    ##############################################################
    # 4.2  Build DataFrame for VWAP feature engineering           #
    ##############################################################
    list_names = data["list_of_data"]
    idx = {n: list_names.index(n) for n in ["close", "high", "low", "volume"]}
    raw = data["raw_data"]

    df = pd.DataFrame({
        "stock_id": data["si"],
        "day_idx":  data["di"],
        "close":    raw[:, idx["close"]],
        "high":     raw[:, idx["high"]],
        "low":      raw[:, idx["low"]],
        "volume":   raw[:, idx["volume"]],
    })
    df = compute_vwap_feats(df)

    ##############################################################
    # 4.3  Determine candidate stock_ids (< max_points samples)   #
    ##############################################################
    counts = Counter(data["si"])
    large_stocks = [sid for sid, c in counts.items() if c >= args.max_points]
    if not large_stocks:
        print(f"No stocks with < {args.max_points} points – nothing to do.")
        sys.exit(0)

    print(f"Found {len(large_stocks)} stocks with >= {args.max_points} rows.")

    ##############################################################
    # 4.4  Train per stock and stream-write JSONL                 #
    ##############################################################
    out_path = Path(args.out)
    mode = "a" if args.append else "w"
    with out_path.open(mode) as fh:
        for sid in sorted(large_stocks):
            mask = (data["si"] == sid)
            result = train_one_stock(sid, mask, data, df, args)
            if result is not None:
                fh.write(json.dumps(result) + "\n")
                print(f"[{sid:>5}]  done – hybrid R² = {result['r2_hybrid']:+.4f}")
    print(f"\nAll finished. Consolidated results → {out_path.resolve()}")


if __name__ == "__main__":
    main()
