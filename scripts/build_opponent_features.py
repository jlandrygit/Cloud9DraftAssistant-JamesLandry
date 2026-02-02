"""Build opponent-facing feature matrices for punishment modeling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import LabelEncoder

from ml.player_comfort_model import get_player_comfort


def main() -> None:
    parser = argparse.ArgumentParser(description="Build opponent feature matrices.")
    parser.add_argument(
        "--input",
        default="data/processed/player_vs_opponent_champion_stats.csv",
        help="Path to player_vs_opponent_champion_stats dataset",
    )
    parser.add_argument(
        "--output",
        default="data/processed",
        help="Output directory for feature artifacts",
    )
    parser.add_argument(
        "--role",
        default="UNKNOWN",
        help="Role value used for comfort lookup when role is unavailable",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    required = {"player_name", "champion_name", "opponent_team_name", "games_played", "win_rate"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.sort_values(["player_name", "champion_name", "opponent_team_name"]).reset_index(drop=True)

    player_encoder = LabelEncoder()
    champion_encoder = LabelEncoder()
    opponent_encoder = LabelEncoder()
    player_encoder.fit(df["player_name"].astype(str))
    champion_encoder.fit(df["champion_name"].astype(str))
    opponent_encoder.fit(df["opponent_team_name"].astype(str))

    player_idx = player_encoder.transform(df["player_name"].astype(str))
    champion_idx = champion_encoder.transform(df["champion_name"].astype(str))
    opponent_idx = opponent_encoder.transform(df["opponent_team_name"].astype(str))

    n_rows = len(df)
    player_oh = sparse.csr_matrix(
        (np.ones(n_rows), (np.arange(n_rows), player_idx)),
        shape=(n_rows, len(player_encoder.classes_)),
    )
    champion_oh = sparse.csr_matrix(
        (np.ones(n_rows), (np.arange(n_rows), champion_idx)),
        shape=(n_rows, len(champion_encoder.classes_)),
    )
    opponent_oh = sparse.csr_matrix(
        (np.ones(n_rows), (np.arange(n_rows), opponent_idx)),
        shape=(n_rows, len(opponent_encoder.classes_)),
    )

    # Model 1 comfort score enriches matchup features.
    comfort_scores = np.array(
        [
            get_player_comfort(player, args.role, champion, fallback="heuristic")
            for player, champion in zip(df["player_name"], df["champion_name"])
        ],
        dtype=float,
    )

    win_rate = df["win_rate"].astype(float).to_numpy()
    games_played = df["games_played"].astype(float).to_numpy()
    # Sample size confidence caps at 1.0 for stability on large histories.
    sample_confidence = np.minimum(1.0, games_played / 20.0)

    numeric = np.vstack([comfort_scores, win_rate, sample_confidence]).T
    numeric_sparse = sparse.csr_matrix(numeric)

    X = sparse.hstack([player_oh, champion_oh, opponent_oh, numeric_sparse], format="csr")

    sparse.save_npz(output_dir / "X_opponent.npz", X)
    metadata = {
        "feature_blocks": ["player_onehot", "champion_onehot", "opponent_onehot", "numeric"],
        "numeric_features": ["player_comfort", "historical_win_rate", "sample_confidence"],
        "row_count": n_rows,
    }
    (output_dir / "opponent_feature_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    Path(output_dir / "player_encoder.pkl").write_bytes(joblib_dump_bytes(player_encoder))
    Path(output_dir / "champion_encoder.pkl").write_bytes(joblib_dump_bytes(champion_encoder))
    Path(output_dir / "opponent_team_encoder.pkl").write_bytes(joblib_dump_bytes(opponent_encoder))


def joblib_dump_bytes(obj) -> bytes:
    """Serialize an object to bytes using joblib."""
    import joblib
    from io import BytesIO

    buffer = BytesIO()
    joblib.dump(obj, buffer)
    return buffer.getvalue()


if __name__ == "__main__":
    main()
