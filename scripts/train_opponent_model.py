"""Train an opponent risk classifier with calibration diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
from scipy import sparse
from sklearn.calibration import calibration_curve
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split


def main() -> None:
    parser = argparse.ArgumentParser(description="Train opponent risk model.")
    parser.add_argument(
        "--features",
        default="data/processed/X_opponent.npz",
        help="Path to opponent feature matrix",
    )
    parser.add_argument(
        "--labels",
        default="data/processed/y_opponent.npy",
        help="Path to opponent labels",
    )
    parser.add_argument(
        "--labels-from-csv",
        default="data/processed/player_vs_opponent_champion_stats.csv",
        help="Fallback CSV for deriving labels when y_opponent.npy is missing",
    )
    parser.add_argument(
        "--label-threshold",
        type=float,
        default=0.5,
        help="Win-rate threshold to derive binary loss labels",
    )
    parser.add_argument(
        "--output",
        default="models/opponent_risk_model.pkl",
        help="Output model path",
    )
    args = parser.parse_args()

    X = sparse.load_npz(args.features)
    labels_path = Path(args.labels)
    if labels_path.exists():
        y = np.load(labels_path).astype(int)
    else:
        csv_path = Path(args.labels_from_csv)
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Missing labels file {labels_path} and fallback CSV {csv_path}."
            )
        # Derive binary loss labels from win_rate when explicit labels are missing.
        import pandas as pd

        df = pd.read_csv(csv_path)
        if "win_rate" not in df.columns:
            raise ValueError("Fallback CSV missing win_rate column for labels.")
        y = (df["win_rate"].astype(float) < args.label_threshold).astype(int).to_numpy()

    # Handle class imbalance via sample weights.
    pos = y.sum()
    neg = len(y) - pos
    weight_pos = (neg / pos) if pos else 1.0
    sample_weight = np.where(y == 1, weight_pos, 1.0)

    x_train, x_val, y_train, y_val, w_train, w_val = train_test_split(
        X, y, sample_weight, test_size=0.2, random_state=42, stratify=y
    )

    model = GradientBoostingClassifier(random_state=42)
    model.fit(x_train.toarray(), y_train, sample_weight=w_train)

    probas = model.predict_proba(x_val.toarray())[:, 1]
    auc = roc_auc_score(y_val, probas)
    brier = brier_score_loss(y_val, probas)
    print(f"Validation AUC: {auc:.4f}")
    print(f"Validation Brier score: {brier:.4f}")

    frac_pos, mean_pred = calibration_curve(y_val, probas, n_bins=10, strategy="uniform")
    print("Calibration curve (mean_pred, frac_pos):")
    for mp, fp in zip(mean_pred, frac_pos):
        print(f"{mp:.3f}\t{fp:.3f}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)


if __name__ == "__main__":
    main()
