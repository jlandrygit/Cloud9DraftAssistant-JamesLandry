"""Train a baseline model to estimate loss risk after opponent picks a champion."""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd
from packaging.version import Version
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

TARGET_COLUMN = "loss"
OUTPUT_PATH = Path("models/opponent_punish_model.pkl")
FALLBACK_CSV = Path("data/processed/player_vs_opponent_champion_stats.csv")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


def _parse_patch(value: str) -> float:
    """Convert patch strings into sortable numeric values."""
    try:
        version = Version(str(value))
        return float(f"{version.major}.{version.minor}")
    except Exception:
        return 0.0


def train_model(data_path: str) -> None:
    """Train and serialize a calibrated logistic regression model."""
    path = Path(data_path)
    if path.exists():
        data = pd.read_parquet(data_path) if data_path.endswith(".parquet") else pd.read_csv(data_path)
    else:
        data = _build_fallback_features(FALLBACK_CSV)

    required = {"opponent_comfort", "global_winrate", "patch", "side", TARGET_COLUMN}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # This model is optional but valuable as a quick, interpretable baseline
    # that estimates punish risk without needing a deep ML stack.
    data["patch"] = data["patch"].map(_parse_patch)
    latest_patch = float(data["patch"].max())
    data = data[data["patch"] == latest_patch]
    features = data[["opponent_comfort", "global_winrate", "patch", "side"]].copy()
    features = pd.get_dummies(features, columns=["side"], drop_first=True)

    target = data[TARGET_COLUMN].astype(int)
    x_train, x_val, y_train, y_val = train_test_split(
        features, target, test_size=0.2, random_state=42, stratify=target
    )

    # Logistic regression provides calibrated probabilities after fitting.
    base_model = LogisticRegression(max_iter=1000)
    model = CalibratedClassifierCV(base_model, method="sigmoid")
    model.fit(x_train, y_train)

    probs = model.predict_proba(x_val)[:, 1]
    auc = roc_auc_score(y_val, probs)
    print(f"Validation ROC AUC: {auc:.4f}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, OUTPUT_PATH)


def _build_fallback_features(csv_path: Path) -> pd.DataFrame:
    """Build minimal opponent punish features from aggregated stats."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing fallback CSV {csv_path}.")
    df = pd.read_csv(csv_path)
    required = {"player_name", "champion_name", "win_rate"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Fallback CSV missing columns: {sorted(missing)}")
    # Use comfort model (heuristic fallback) for opponent comfort.
    from ml.player_comfort_model import get_player_comfort

    opponent_comfort = [
        get_player_comfort(str(player), "UNKNOWN", str(champion)) or 0.0
        for player, champion in zip(df["player_name"], df["champion_name"])
    ]
    data = pd.DataFrame(
        {
            "opponent_comfort": opponent_comfort,
            "global_winrate": df["win_rate"].astype(float),
            "patch": "unknown",
            "side": "BLUE",
        }
    )
    data[TARGET_COLUMN] = (df["win_rate"].astype(float) < 0.5).astype(int)
    return data


if __name__ == "__main__":
    train_model("data/processed/opponent_punish_features.parquet")
