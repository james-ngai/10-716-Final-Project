# Project Overview

This repository contains the complete codebase for training, evaluating, and analyzing multiple financial prediction models based on alpha signals and ARIMA-residual techniques.

## Folder Structure

### `final_models/`
- Contains model definitions and training scripts for each architecture (e.g., ARIMA + LSTM, ARIMA + Transformer hybrids).
- Each script includes instructions in the comment header detailing how to run the training.

**Example usage:**
```bash
python final_models/arima_lstm_novwap.py --data ./data/dataset.npy --out results/arima_lstm_novwap.jsonl
```

### `post_processing/`
- `vif.py`: Tests for multicollinearity among alpha signals using Variance Inflation Factor (VIF) analysis.
- Scripts for extracting ARIMA model results and fixing related extraction issues.

### `results/`
- `.jsonl` files containing model evaluation outputs on the test dataset.
- Each line represents the results for a single stock, including metrics like MSE, MAE, and R².

### `plots/`
- Plots generated from statistics computed over model results.
- Includes return curves, Sharpe ratio distributions, and model comparisons.

### `statistics/`
- Scripts to aggregate model results.
- Compute weighted metrics and generate analytical plots.

## How to Train a Model

To train a model:
1. Navigate to the `final_models/` directory.
2. Choose the desired model script.
3. Follow the command format specified in the script's comment header.

Training scripts usually require:
- A `.npy` data file.
- An output path for storing `.jsonl` results.

## Example Training Command
```bash
python final_models/arima_transformer_vwap.py --data ./data/alpha_signals.npy --out results/arima_transformer_vwap.jsonl
```

## Notes
- **Data** should be preprocessed into a `.npy` file as expected by the training scripts.
- **Alpha signals** can be pruned using `post_processing/vif.py` to reduce multicollinearity.
- **ARIMA** model results extraction issues are handled with dedicated scripts inside `post_processing/`.