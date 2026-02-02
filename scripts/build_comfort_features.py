"""Build model-ready comfort features and labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def main() -> None:
    parser = argparse.ArgumentParser(description="Build comfort feature arrays.")
    parser.add_argument(
        "--input",
        default="data/processed/player_champion_stats.parquet",
        help="Path to player_champion_stats dataset",
    )
    parser.add_argument(
        "--output",
        default="data/processed",
        help="Output directory for numpy arrays",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(input_path) if input_path.suffix == ".parquet" else pd.read_csv(input_path)

    required = {"player_name", "champion_name", "role", "games_played", "win_rate", "recency_weighted_games"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.sort_values(["player_name", "role", "champion_name"]).reset_index(drop=True)

    df["games_played_log"] = np.log1p(df["games_played"].astype(float))

    # Role consistency: share of games for this role among the player's total games.
    role_totals = df.groupby(["player_name", "role"])["games_played"].sum().rename("role_games")
    player_totals = df.groupby("player_name")["games_played"].sum().rename("player_games")
    df = df.join(role_totals, on=["player_name", "role"]).join(player_totals, on="player_name")
    df["role_consistency"] = df["role_games"] / df["player_games"]

    feature_columns = ["games_played_log", "win_rate", "recency_weighted_games", "role_consistency"]
    X = df[feature_columns].astype(float).to_numpy()
    y = df["comfort_score"].astype(float).to_numpy() if "comfort_score" in df.columns else np.zeros(len(df))

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    np.save(output_dir / "X_comfort.npy", X_scaled)
    np.save(output_dir / "y_comfort.npy", y)

    metadata = {
        "feature_columns": feature_columns,
        "row_count": len(df),
        "ordering": ["player_name", "role", "champion_name"],
    }
    (output_dir / "comfort_feature_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
