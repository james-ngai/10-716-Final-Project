#!/usr/bin/env python3

import json
import pathlib

# ----------- Config -------------
JSONL_FILES = [
    "arima_baseline.jsonl",
    "arima_lstm_novwap.jsonl",
    "arima_lstm_vwap.jsonl",
    "arima_transformer_novwap.jsonl",
    "arima_transformer_vwap.jsonl",
]

JSONL_FILES = [f"results/{file}" for file in JSONL_FILES]

# ---------------------------------

def compute_weighted_mse(jsonl_path):
    total_weighted_mse = 0.0
    total_points = 0

    with open(jsonl_path, 'r') as f:
        for line in f:
            entry = json.loads(line)
            n_points = entry.get('n_points')
            if 'mse_hybrid' in entry:
                mse_hybrid = entry.get('mse_hybrid')
            else:
                mse_hybrid = entry.get('mse_arima')
            if n_points is None or mse_hybrid is None:
                continue  # skip if missing fields

            total_weighted_mse += n_points * mse_hybrid
            total_points += n_points

    if total_points == 0:
        raise ValueError(f"No valid points found in {jsonl_path}.")

    combined_mse = total_weighted_mse / total_points
    return combined_mse

if __name__ == "__main__":
    for file_path in JSONL_FILES:
        jsonl_path = pathlib.Path(file_path)
        try:
            combined_mse = compute_weighted_mse(jsonl_path)
            print(f"{jsonl_path.name}: Combined Weighted MSE = {combined_mse:.6f}")
        except Exception as e:
            print(f"Error processing {jsonl_path.name}: {e}")
