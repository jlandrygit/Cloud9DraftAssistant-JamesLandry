"""Training pipeline for draft evaluation models."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd
import xgboost as xgb
from packaging.version import Version
from sklearn.metrics import classification_report, roc_auc_score

LABEL_COLUMN = "label"
PATCH_COLUMN = "patch"


def train_draft_evaluator(
    data_path: str | Path,
    model_path: str | Path,
    *,
    file_format: Literal["csv", "parquet"] | None = None,
    test_patch: str | None = None,
) -> dict[str, float]:
    """Train an XGBoost classifier to predict win/loss from draft features.

    The dataset is split by patch to avoid temporal leakage: earlier patches
    are used for training and later patches for testing.
    """
    path = Path(data_path)
    if file_format is None:
        file_format = "parquet" if path.suffix == ".parquet" else "csv"

    data = _load_dataset(path, file_format)
    _validate_columns(data)

    train_df, test_df = _split_by_patch(data, test_patch=test_patch)

    x_train, y_train = _split_features_labels(train_df)
    x_test, y_test = _split_features_labels(test_df)

    # Handle class imbalance via scale_pos_weight (neg / pos).
    pos = y_train.sum()
    neg = len(y_train) - pos
    scale_pos_weight = (neg / pos) if pos > 0 else 1.0

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        n_jobs=4,
        random_state=42,
    )

    model.fit(x_train, y_train)

    preds = model.predict(x_test)
    probas = model.predict_proba(x_test)[:, 1]
    metrics = {
        "roc_auc": roc_auc_score(y_test, probas),
        "accuracy": float((preds == y_test).mean()),
    }

    print(classification_report(y_test, preds, digits=4))

    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(model_path))

    return metrics


def _load_dataset(path: Path, file_format: Literal["csv", "parquet"]) -> pd.DataFrame:
    """Load features from disk."""
    if file_format == "parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _validate_columns(df: pd.DataFrame) -> None:
    """Ensure required columns exist."""
    missing = {LABEL_COLUMN, PATCH_COLUMN} - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")


def _split_by_patch(
    df: pd.DataFrame, *, test_patch: str | None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data by patch, training on earlier patches."""
    patches = sorted(df[PATCH_COLUMN].dropna().unique(), key=_parse_patch)
    if not patches:
        raise ValueError("No patch values available for split.")

    target_patch = test_patch or patches[-1]
    if target_patch not in patches:
        raise ValueError(f"Patch {target_patch} not found in data.")

    train_df = df[df[PATCH_COLUMN].map(_parse_patch) < _parse_patch(target_patch)]
    test_df = df[df[PATCH_COLUMN] == target_patch]
    if train_df.empty or test_df.empty:
        raise ValueError("Patch split resulted in empty train/test set.")
    return train_df, test_df


def _split_features_labels(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate feature columns from labels."""
    features = df.drop(columns=[LABEL_COLUMN, PATCH_COLUMN])
    labels = df[LABEL_COLUMN].astype(int)
    return features, labels


def _parse_patch(patch: str) -> Version:
    """Parse a patch string into a comparable semantic version."""
    return Version(str(patch))
