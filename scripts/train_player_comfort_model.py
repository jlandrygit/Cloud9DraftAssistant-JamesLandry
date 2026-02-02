"""Train a regression model to approximate heuristic comfort scores."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import joblib
import pandas as pd
from packaging.version import Version
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

TARGET_COLUMN = "comfort_score"
PATCH_COLUMN = "patch"
RECENCY_COLUMN = "recency_weighted_games"
GAMES_COLUMN = "games_played"
WINRATE_COLUMN = "win_rate"
PLAYER_COLUMN = "player_name"
ROLE_COLUMN = "role"
CHAMPION_COLUMN = "champion_name"


def _parse_patch(value: str) -> Version:
    """Parse patch strings into sortable semantic versions."""
    try:
        return Version(str(value))
    except Exception:
        return Version("0")


def _select_numeric_features(df: pd.DataFrame, drop: Iterable[str]) -> pd.DataFrame:
    """Select numeric columns and drop identifiers/targets."""
    numeric = df.select_dtypes(include="number").copy()
    return numeric.drop(columns=[col for col in drop if col in numeric.columns])


def train_model(data_path: str, output_path: str) -> None:
    """Train and serialize a player comfort regression model."""
    data = pd.read_parquet(data_path) if data_path.endswith(".parquet") else pd.read_csv(data_path)
    # Regression is used because the heuristic comfort score is continuous.
    if TARGET_COLUMN not in data.columns:
        data[TARGET_COLUMN] = _compute_heuristic_comfort(data)

    if PATCH_COLUMN in data.columns:
        patches = sorted(data[PATCH_COLUMN].dropna().unique(), key=_parse_patch)
        if not patches:
            raise ValueError("No patch values available for split.")
        latest_patch = patches[-1]
        recent_df = data[data[PATCH_COLUMN] == latest_patch]
    else:
        recent_df = data

    if recent_df.empty:
        raise ValueError("Latest patch subset is empty.")

    train_df, val_df = train_test_split(
        recent_df, test_size=0.2, random_state=42, shuffle=True
    )

    feature_df = _select_numeric_features(data, drop=[TARGET_COLUMN])
    features = feature_df.columns

    x_train = _select_numeric_features(train_df, drop=[TARGET_COLUMN])
    y_train = train_df[TARGET_COLUMN].astype(float)
    x_val = _select_numeric_features(val_df, drop=[TARGET_COLUMN])
    y_val = val_df[TARGET_COLUMN].astype(float)

    model = XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        n_jobs=4,
        random_state=42,
    )
    model.fit(x_train, y_train)

    # Quantitative check: RMSE on the patch-held validation set.
    val_predictions = model.predict(x_val)
    rmse = mean_squared_error(y_val, val_predictions) ** 0.5
    print(f"Validation RMSE: {rmse:.4f}")

    # Feature importance helps interpret which signals drive comfort.
    importance = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    print("Top feature importances:")
    print(importance.head(20).to_string())

    # Qualitative checks are acceptable here because we want a quick, human-readable
    # sanity check of comfort rankings without overfitting to a numeric target.
    _print_player_rankings(val_df, model, n_players=5)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output)


def _print_player_rankings(
    val_df: pd.DataFrame, model: XGBRegressor, n_players: int = 5
) -> None:
    """Print top predicted champions and compare to historical frequency."""
    players = val_df.get(PLAYER_COLUMN, pd.Series(dtype=str)).dropna().unique().tolist()
    if not players:
        print("No player column available for qualitative checks.")
        return

    sample_players = pd.Series(players).sample(
        n=min(n_players, len(players)), random_state=42
    )
    for player in sample_players:
        player_df = val_df[val_df[PLAYER_COLUMN] == player].copy()
        if player_df.empty or CHAMPION_COLUMN not in player_df.columns:
            continue

        feature_df = _select_numeric_features(
            player_df, drop=[TARGET_COLUMN]
        )
        player_df["predicted_comfort"] = model.predict(feature_df)

        top_pred = (
            player_df.sort_values("predicted_comfort", ascending=False)
            .drop_duplicates(subset=[CHAMPION_COLUMN])
            .head(5)
        )
        print(f"\nPlayer: {player}")
        print("Top 5 predicted comfort champions:")
        print(top_pred[[CHAMPION_COLUMN, "predicted_comfort"]].to_string(index=False))

        # Compare to historical pick frequency by simple count.
        freq = (
            player_df[CHAMPION_COLUMN]
            .value_counts()
            .head(5)
            .rename_axis(CHAMPION_COLUMN)
            .reset_index(name=GAMES_COLUMN)
        )
        print("Top 5 historical picks:")
        print(freq.to_string(index=False))


def _compute_heuristic_comfort(df: pd.DataFrame) -> pd.Series:
    """Compute a fallback comfort score from available stats.

    We only care about how often a player has piloted the champion in-role.
    """
    for col in [PLAYER_COLUMN, ROLE_COLUMN, GAMES_COLUMN]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    result = df.copy()
    role_totals = result.groupby([PLAYER_COLUMN, ROLE_COLUMN])[GAMES_COLUMN].transform(
        "sum"
    )
    games_pct = result[GAMES_COLUMN].astype(float) / role_totals.replace(0, 1.0)
    # Already normalized to [0, 1] within player-role.
    return games_pct.clip(lower=0.0, upper=1.0)


if __name__ == "__main__":
    train_model(
        data_path="data/processed/player_champion_stats.parquet",
        output_path="models/player_comfort_model.pkl",
    )
