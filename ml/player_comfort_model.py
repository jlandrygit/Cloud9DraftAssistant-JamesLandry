"""Player comfort scoring using a trained regression model."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

MODEL_PATH = Path("models/player_comfort_model.pkl")
FEATURES_PATH = Path("data/processed/player_champion_stats.parquet")
TARGET_COLUMN = "comfort_score"


@lru_cache(maxsize=1)
def _load_model():
    """Load the trained comfort model from disk."""
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


@lru_cache(maxsize=1)
def _load_features() -> pd.DataFrame:
    """Load precomputed comfort features from disk."""
    if FEATURES_PATH.suffix == ".parquet":
        return pd.read_parquet(FEATURES_PATH)
    return pd.read_csv(FEATURES_PATH)


@lru_cache(maxsize=1)
def _load_role_totals() -> pd.Series | None:
    """Precompute total games per player-role for heuristic fallback."""
    features = _load_features()
    if "games_played" not in features.columns:
        return None
    return features.groupby(["player_name", "role"])["games_played"].sum()


def get_player_comfort(
    player: str, role: str, champion: str, fallback: str = "heuristic"
) -> float | None:
    """Return a comfort score in [0, 1] for a player/champion/role.

    Returns None if the player has no historical data.
    Warning: this model estimates player comfort, not win probability.
    """
    features = _load_features()
    row = features[
        (features["player_name"] == player)
        & (features["role"] == role)
        & (features["champion_name"] == champion)
    ]
    if row.empty:
        return None

    if fallback == "heuristic" and TARGET_COLUMN in row.columns:
        return _clamp(float(row[TARGET_COLUMN].iloc[0]))

    model = _load_model()
    if model is None:
        if TARGET_COLUMN in row.columns:
            return _clamp(float(row[TARGET_COLUMN].iloc[0]))
        if "games_played" in row.columns:
            totals = _load_role_totals()
            total_games = (
                float(totals.get((player, role), 0.0)) if totals is not None else 0.0
            )
            if total_games <= 0:
                return 0.0
            return _clamp(float(row["games_played"].iloc[0]) / total_games)
        return 0.0
    numeric = row.select_dtypes(include="number").drop(
        columns=[col for col in [TARGET_COLUMN] if col in row.columns]
    )
    if numeric.empty:
        if TARGET_COLUMN in row.columns:
            return _clamp(float(row[TARGET_COLUMN].iloc[0]))
        if "games_played" in row.columns:
            totals = _load_role_totals()
            total_games = (
                float(totals.get((player, role), 0.0)) if totals is not None else 0.0
            )
            if total_games <= 0:
                return 0.0
            return _clamp(float(row["games_played"].iloc[0]) / total_games)
        return 0.0

    prediction = float(model.predict(numeric)[0])
    return _clamp(prediction)


def _clamp(value: float) -> float:
    """Clamp a score to [0, 1]."""
    return max(0.0, min(1.0, value))
