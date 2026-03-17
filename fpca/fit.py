"""Fit the FPCA population model from training data.

Reads the GoldenCheetah training dataset, fits PCA on log-transformed
power-duration curves, and writes the model artifact to src/silhouette/.

Usage:
    python fpca/fit.py

See fpca/README.md for details.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

FPCA_DIR = Path(__file__).parent
OUTPUT_PATH = FPCA_DIR.parent / "src" / "silhouette" / "_fpca_model.npz"

N_TRAIN = 2000
N_COMPONENTS = 3
RANDOM_STATE = 123


def main():
    # Load data
    data = pd.read_csv(FPCA_DIR / "GCdatcrop3.csv")
    power = data.iloc[:, 3:93].dropna().values

    time_grid = pd.read_csv(FPCA_DIR / "GCdat2time.csv").iloc[:, 0].values

    print(f"Loaded {power.shape[0]} athletes, {power.shape[1]} time points")

    # Train/test split (matching the original R analysis)
    rng = np.random.RandomState(RANDOM_STATE)
    train_idx = rng.choice(len(power), size=N_TRAIN, replace=False)
    train = power[train_idx]

    print(f"Training set: {train.shape[0]} athletes")

    # Log-transform
    log_train = np.log(train)

    # Fit PCA
    mean_function = log_train.mean(axis=0)
    centered = log_train - mean_function

    pca = PCA(n_components=N_COMPONENTS)
    pc_scores = pca.fit_transform(centered)

    # Save
    np.savez(
        OUTPUT_PATH,
        mean_function=mean_function,
        eigenfunctions=pca.components_.T,
        eigenvalues=pca.explained_variance_,
        explained_variance_ratio=pca.explained_variance_ratio_,
        pc_scores=pc_scores,
        time_grid=time_grid,
    )

    print(f"\nVariance explained ({N_COMPONENTS} components):")
    for i, ratio in enumerate(pca.explained_variance_ratio_):
        print(f"  FPC{i + 1}: {ratio * 100:.1f}%")
    print(f"  Total: {pca.explained_variance_ratio_.sum() * 100:.1f}%")

    print(f"\nModel written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
