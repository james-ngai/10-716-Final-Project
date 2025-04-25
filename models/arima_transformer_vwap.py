#!/usr/bin/env python3
# arima_transformer_residual_batch.py
#
# Train the ARIMA-residual + Transformer hybrid on **all stocks that have at
# least MIN_POINTS observations** and write each result as one line of JSON
# to a single JSONL file.
#
# Example
# -------
"""
python3 final_tests/arima_transformer_vwap.py \
    --data  ./data/dict_of_data_Jan2025_part1.npy \
    --out results/testing_arima_transformer_vwap.jsonl
"""
#
# ---------------------------------------------------------------------------

import argparse, json, math, sys, warnings
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

def _py(x):
    """Convert a NumPy scalar to a plain Python scalar, else return as-is."""
    if isinstance(x, (np.generic,)):   # catches np.int64, np.float32, ...
        return x.item()
    return x


# --------------------------------------------------------------------------- #
# 1.  CLI                                                                      #
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Batch ARIMA + Transformer residual hybrid")
    # data
    p.add_argument("--data",   required=True, type=str)
    p.add_argument("--data2",  default=None,  type=str)
    # selection
    p.add_argument("--min-points", type=int, default=500,
                   help="Train only on stocks with ≥ THIS many rows")
    # window / training
    p.add_argument("--window",  type=int,   default=10)
    p.add_argument("--epochs",  type=int,   default=40)
    p.add_argument("--batch",   type=int,   default=256)
    p.add_argument("--lr",      type=float, default=1e-3)
    p.add_argument("--split",   type=float, default=0.8)
    # Transformer hyper-params
    p.add_argument("--d-model", type=int,   default=256)
    p.add_argument("--nhead",   type=int,   default=4)
    p.add_argument("--layers",  type=int,   default=3)
    p.add_argument("--ff",      type=int,   default=256)
    p.add_argument("--dropout", type=float, default=0.1)
    # output
    p.add_argument("--out",     type=str,   default="transformer_big.jsonl")
    p.add_argument("--append",  action="store_true",
                   help="Append to existing --out instead of overwriting")
    return p.parse_args()


# --------------------------------------------------------------------------- #
# 2.  Dataset utilities                                                       #
# --------------------------------------------------------------------------- #
class ResidualDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, seq: int):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))
        self.seq = seq
    # guard against negative length
    def __len__(self): return max(0, len(self.X) - self.seq)
    def __getitem__(self, idx):
        return self.X[idx:idx+self.seq], self.y[idx + self.seq]


# --------------------------------------------------------------------------- #
# 3.  Transformer model                                                       #
# --------------------------------------------------------------------------- #
def positional_encoding(seq_len: int, d_model: int, device):
    pe = torch.zeros(seq_len, d_model, device=device)
    position = torch.arange(seq_len, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, device=device) *
                         (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)                # (1, T, D)

class ResidualTransformer(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, ff, dropout):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model, nhead,
                                               ff, dropout,
                                               batch_first=True,
                                               activation="gelu")
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        B, T, _ = x.shape
        h = self.proj(x) + positional_encoding(T, h_dim := self.proj.out_features, x.device)
        h = self.encoder(h)
        return self.fc(h[:, -1]).squeeze(-1)


# --------------------------------------------------------------------------- #
# 4.  Helpers: ARIMA forecasts & VWAP                                         #
# --------------------------------------------------------------------------- #
def rolling_arima(series: np.ndarray, order=(1,1,1)) -> np.ndarray:
    p, d, q = order
    min_hist = max(p, d, q) + 1
    preds = np.full_like(series, np.nan, dtype=np.float32)
    hist = list(series[:min_hist])
    for t in range(min_hist, len(series)):
        try:
            preds[t] = float(ARIMA(hist, order=order).fit().forecast()[0])
        except Exception:
            preds[t] = hist[-1]
        hist.append(series[t])
    return preds

def build_vwap_df(data: dict) -> pd.DataFrame:
    names = data["list_of_data"]
    idx = {n: names.index(n) for n in ("close", "high", "low")}
    raw = data["raw_data"]
    df = pd.DataFrame({
        "stock_id": data["si"],
        "day_idx":  data["di"],
        "close":    raw[:, idx["close"]],
        "high":     raw[:, idx["high"]],
        "low":      raw[:, idx["low"]],
    })
    df["vwap"] = (df["high"] + df["low"] + df["close"])/3
    df.sort_values(["stock_id", "day_idx"], inplace=True)
    df["vwap_5"] = df.groupby("stock_id")["vwap"].transform(lambda x: x.rolling(5,1).mean())
    df["dev_vwap"] = (df["close"] - df["vwap"]) / df["vwap"]
    df["trend_flag"] = (df["close"] > df["vwap"]).astype(np.float32)
    return df


# --------------------------------------------------------------------------- #
# 5.  Per-stock routine                                                       #
# --------------------------------------------------------------------------- #
def train_one_stock(sid: int, mask: np.ndarray, data: dict, vwap_df: pd.DataFrame,
                    args: argparse.Namespace) -> dict | None:
    X_raw  = data["x_data"][mask]      # (T, 200)
    y_all  = data["y_data"][mask]
    vwap_s = vwap_df.loc[mask, ["vwap_5", "dev_vwap", "trend_flag"]].to_numpy(np.float32)

    # ARIMA forecasts
    arima_all = rolling_arima(y_all)
    good = ~np.isnan(arima_all)
    if good.sum() < args.window + 5:
        print(f"[{sid:>5}] skipped – too few usable samples ({good.sum()})")
        return None

    X_raw, y_all, vwap_s, arima_all = [arr[good] for arr in
                                       (X_raw, y_all, vwap_s, arima_all)]
    X_aug = np.concatenate([X_raw, vwap_s, arima_all[:,None]], axis=1)  # (T', 204)

    # split
    split = int(len(y_all) * args.split)
    X_tr, X_te = X_aug[:split], X_aug[split:]
    y_tr, y_te = y_all[:split], y_all[split:]
    arima_tr, arima_te = arima_all[:split], arima_all[split:]
    resid_tr, resid_te = y_tr - arima_tr, y_te - arima_te

    seq = args.window
    ds_tr, ds_te = ResidualDataset(X_tr, resid_tr, seq), ResidualDataset(X_te, resid_te, seq)
    if len(ds_tr)==0 or len(ds_te)==0:
        print(f"[{sid:>5}] skipped – window longer than split sets")
        return None
    dl_tr = DataLoader(ds_tr, batch_size=args.batch, shuffle=True)
    dl_te = DataLoader(ds_te, batch_size=args.batch)

    # model
    net = ResidualTransformer(X_aug.shape[1], args.d_model, args.nhead,
                              args.layers, args.ff, args.dropout).to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    for _ in range(args.epochs):
        net.train()
        for xb, yb in dl_tr:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(); loss = loss_fn(net(xb), yb); loss.backward(); opt.step()

    # predict
    net.eval(); resid_pred = []
    with torch.no_grad():
        for xb, _ in dl_te:
            resid_pred.append(net(xb.to(DEVICE)).cpu())
    resid_pred = torch.cat(resid_pred).numpy()

    skip = seq
    y_te_cut, arima_cut = y_te[skip:], arima_te[skip:]
    resid_true, hybrid = resid_te[skip:], arima_cut + resid_pred

    res = {
        "stock_id": _py(sid),
        "n_points": _py(mask.sum()),
        "mse_arima":  float(mean_squared_error(y_te_cut, arima_cut)),
        "mae_arima":  float(mean_absolute_error(y_te_cut, arima_cut)),
        "mse_hybrid": float(mean_squared_error(y_te_cut, hybrid)),
        "mae_hybrid": float(mean_absolute_error(y_te_cut, hybrid)),
        "r2_hybrid":  float(r2_score(y_te_cut, hybrid)),
        "hyperparams": vars(args),
        "y_test":     y_te_cut.tolist(),
        "arima":      arima_cut.tolist(),
        "resid_true": resid_true.tolist(),
        "resid_pred": resid_pred.tolist(),
        "hybrid":     hybrid.tolist(),
        "vwap_cols":  ["vwap_5", "dev_vwap", "trend_flag"],
    }
    return res


# --------------------------------------------------------------------------- #
# 6.  Main driver                                                             #
# --------------------------------------------------------------------------- #
def main() -> None:
    args = parse_args()

    # load & merge
    d1 = np.load(args.data, allow_pickle=True).item()
    data = d1 if args.data2 is None else {k: np.concatenate([d1[k],
                 np.load(args.data2, allow_pickle=True).item()[k]]) for k in d1}

    # VWAP dataframe (same order as original arrays)
    vwap_df = build_vwap_df(data)

    # choose candidate stocks
    counts = Counter(data["si"])
    candidates = [sid for sid, c in counts.items() if c >= args.min_points]
    if not candidates:
        print(f"No stocks with ≥ {args.min_points} rows.")
        sys.exit(0)
    print(f"Found {len(candidates)} stocks with ≥ {args.min_points} rows.")

    out_path = Path(args.out)
    mode = "a" if args.append else "w"
    with out_path.open(mode) as fh:
        for sid in sorted(candidates):
            mask = (data["si"] == sid)
            res = train_one_stock(sid, mask, data, vwap_df, args)
            if res is not None:
                fh.write(json.dumps(res) + "\n")
                print(f"[{sid:>5}] done – hybrid R² = {res['r2_hybrid']:+.4f}")

    print(f"\nFinished. Results written to {out_path.resolve()}")

if __name__ == "__main__":
    main()
