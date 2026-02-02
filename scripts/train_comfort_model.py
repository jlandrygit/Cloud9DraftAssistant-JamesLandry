"""Train a comfort scoring regression model with cross-validation."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold


def main() -> None:
    parser = argparse.ArgumentParser(description="Train comfort regression model.")
    parser.add_argument(
        "--features",
        default="data/processed/X_comfort.npy",
        help="Path to feature matrix",
    )
    parser.add_argument(
        "--labels",
        default="data/processed/y_comfort.npy",
        help="Path to target array",
    )
    parser.add_argument(
        "--output",
        default="models/comfort_model.pkl",
        help="Output model path",
    )
    args = parser.parse_args()

    X = np.load(args.features)
    y = np.load(args.labels)

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    rmses: list[float] = []
    spearmans: list[float] = []

    for train_idx, val_idx in cv.split(X):
        model = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42, max_iter=5000)
        model.fit(X[train_idx], y[train_idx])
        preds = model.predict(X[val_idx])
        rmse = float(np.sqrt(mean_squared_error(y[val_idx], preds)))
        rmses.append(rmse)

        corr, _ = spearmanr(y[val_idx], preds)
        spearmans.append(float(corr))

    print(f"CV RMSE: {np.mean(rmses):.4f} ± {np.std(rmses):.4f}")
    print(f"CV Spearman: {np.mean(spearmans):.4f} ± {np.std(spearmans):.4f}")

    final_model = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42, max_iter=5000)
    final_model.fit(X, y)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, output_path)


if __name__ == "__main__":
    main()
