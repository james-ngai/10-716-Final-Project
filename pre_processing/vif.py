# vif_pruning.py
import numpy as np
import pandas as pd
from pathlib import Path
from statsmodels.stats.outliers_influence import variance_inflation_factor
from tqdm import tqdm


def load_alpha_signals(*npy_paths: Path) -> pd.DataFrame:
    """Load one or more Trexquant *.npy* alpha-signal dictionaries."""
    frames = []
    for p in npy_paths:
        data = np.load(p, allow_pickle=True).item()
        # x_data is (N, 200)         – alpha signals
        # si / di tell us which stock–day row it is.
        cols = [f"alpha_{i:03d}" for i in range(data["x_data"].shape[1])]
        df   = pd.DataFrame(data["x_data"], columns=cols)
        df["stock_id"] = data["si"]
        df["day_idx"]  = data["di"]
        frames.append(df)
    return pd.concat(frames, axis=0, ignore_index=True)


def compute_vif_matrix(df: pd.DataFrame) -> pd.Series:
    """Return a Series of VIF values for the columns in *df*."""
    # statsmodels needs a numpy array; add constant to avoid homogeneity warning
    X = df.values
    vif = pd.Series(
        [variance_inflation_factor(X, i) for i in range(X.shape[1])],
        index=df.columns,
        name="VIF",
    )
    return vif.sort_values(ascending=False)


def vif_prune(
    df: pd.DataFrame,
    vif_threshold: float = 10.0,
    min_features: int = 5,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Iteratively drop the feature with the highest VIF until all remaining
    features have VIF ≤ *vif_threshold* or only *min_features* remain.

    Returns
    -------
    pruned_df  : DataFrame containing *only* the kept alpha signals
    history_df : DataFrame logging VIF values at every pruning step
    """
    work_df   = df.copy()
    history   = []

    while True:
        vif = compute_vif_matrix(work_df)
        history.append(vif)

        # Stop if we are within threshold or can't drop any more.
        if vif.iloc[0] <= vif_threshold or work_df.shape[1] <= min_features:
            if verbose:
                print(
                    f"Finished: max VIF={vif.iloc[0]:.2f}, "
                    f"{work_df.shape[1]} features remain."
                )
            break

        # Drop the worst offender and loop.
        bad_feat = vif.index[0]
        if verbose:
            print(f"Dropping '{bad_feat}' (VIF={vif.iloc[0]:.2f})")
        work_df = work_df.drop(columns=bad_feat)

    history_df = pd.concat(history, axis=1).T.reset_index(drop=True)
    return work_df, history_df


if __name__ == "__main__":
    # ---------------------------------------------
    # 1. Change these paths to your dataset files.
    # ---------------------------------------------
    FILE1 = Path("./data/dict_of_data_Jan2025_part1.npy")
    FILE2 = Path("./data/dict_of_data_Jan2025_part2.npy")  # optional

    # ---------------------------------------------
    # 2. Load & prepare dataframe (200 columns).
    #    We ignore stock/day indices for the VIF step.
    # ---------------------------------------------
    df_full = load_alpha_signals(FILE1).filter(like="alpha_")

    # (optional) Standardize features to unit variance—this isn’t required
    # for VIF but makes the iterative process a bit stabler numerically.
    df_std = (df_full - df_full.mean()) / df_full.std(ddof=0)

    # ---------------------------------------------
    # 3. Run iterative VIF pruning.
    # ---------------------------------------------
    pruned_df, vif_history = vif_prune(
        df_std,
        vif_threshold=10.0,
        verbose=True,
    )

    # ---------------------------------------------
    # 4. Persist outputs so you can reuse them.
    # ---------------------------------------------
    pruned_df.to_parquet("alpha_signals_pruned.parquet")
    vif_history.to_csv("vif_history.csv", index=False)

    print(f"Kept {pruned_df.shape[1]} of {df_full.shape[1]} signals.")
    print("Saved: alpha_signals_pruned.parquet  |  vif_history.csv")
