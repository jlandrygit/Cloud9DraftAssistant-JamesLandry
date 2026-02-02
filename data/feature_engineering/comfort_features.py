"""Preprocessing utilities for champion comfort feature inputs."""

from __future__ import annotations

from datetime import datetime

import pandas as pd
from packaging.version import Version

REQUIRED_PATCH_COLUMN = "patch"
REQUIRED_MATCH_DATE_COLUMN = "match_date"
REQUIRED_ROLE_COLUMN = "role"

VALID_ROLES = {"TOP", "JUNGLE", "MID", "ADC", "SUPPORT"}
COMFORT_WEIGHT_GAMES_PCT = 1.0


def normalize_patch_versions(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize patch versions using packaging.version.Version.

    Expected columns:
        - patch: patch identifier (e.g., "14.2", "14.10")
    """
    _ensure_columns(df, [REQUIRED_PATCH_COLUMN])
    result = df.copy()
    result[REQUIRED_PATCH_COLUMN] = result[REQUIRED_PATCH_COLUMN].map(
        lambda value: str(Version(str(value)))
    )
    return result


def add_time_decay(df: pd.DataFrame, half_life_days: int = 90) -> pd.DataFrame:
    """Compute time-decay weights based on match date.

    Expected columns:
        - match_date: date or datetime of the match
    """
    _ensure_columns(df, [REQUIRED_MATCH_DATE_COLUMN])
    result = df.copy()
    match_dates = pd.to_datetime(result[REQUIRED_MATCH_DATE_COLUMN], errors="raise")
    reference_date = match_dates.max()
    age_days = (reference_date - match_dates).dt.days
    result["time_decay_weight"] = 0.5 ** (age_days / float(half_life_days))
    return result


def validate_roles(df: pd.DataFrame) -> pd.DataFrame:
    """Validate role values and enforce role-specific separation.

    Expected columns:
        - role: role label (TOP, JUNGLE, MID, ADC, SUPPORT)
    """
    _ensure_columns(df, [REQUIRED_ROLE_COLUMN])
    result = df.copy()
    invalid = set(result[REQUIRED_ROLE_COLUMN].dropna().unique()) - VALID_ROLES
    if invalid:
        raise ValueError(f"Invalid role values found: {sorted(invalid)}")
    return result


def compute_player_champion_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute player/champion/role stats with time-decay weighting.

    Expected columns:
        - player: player identifier
        - champion: champion identifier
        - role: role label (TOP, JUNGLE, MID, ADC, SUPPORT)
        - win: 1/0 or True/False win indicator
        - match_date: date or datetime of the match
        - time_decay_weight: precomputed decay weight
    """
    _ensure_columns(
        df,
        [
            "player",
            "champion",
            REQUIRED_ROLE_COLUMN,
            "win",
            REQUIRED_MATCH_DATE_COLUMN,
            "time_decay_weight",
        ],
    )
    result = df.copy()
    result[REQUIRED_MATCH_DATE_COLUMN] = pd.to_datetime(
        result[REQUIRED_MATCH_DATE_COLUMN], errors="raise"
    )

    grouped = (
        result.groupby(["player", "champion", REQUIRED_ROLE_COLUMN], as_index=False)
        .agg(
            games_played=("champion", "size"),
            weighted_games=("time_decay_weight", "sum"),
            winrate=("win", "mean"),
            last_played_date=(REQUIRED_MATCH_DATE_COLUMN, "max"),
        )
        .reset_index(drop=True)
    )

    totals = (
        grouped.groupby(["player", REQUIRED_ROLE_COLUMN], as_index=False)
        .agg(role_games_played=("games_played", "sum"))
        .reset_index(drop=True)
    )

    merged = grouped.merge(totals, on=["player", REQUIRED_ROLE_COLUMN], how="left")
    # Normalize by player-role totals so comfort is comparable across players
    # with different sample sizes and role distributions.
    merged["games_played_pct"] = merged["games_played"] / merged["role_games_played"]
    return merged.drop(columns=["role_games_played"])


def compute_heuristic_comfort_score(stats_df: pd.DataFrame) -> pd.DataFrame:
    """Compute a heuristic comfort score from aggregated stats.

    This is a baseline heuristic that only measures how frequently a player
    has piloted a champion in-role. It is useful as a stable fallback when ML
    models are unavailable or for quick sanity checks during data validation.
    """
    _ensure_columns(stats_df, ["games_played_pct"])
    result = stats_df.copy()
    # games_played_pct is already normalized per player-role to [0, 1]
    result["comfort_score"] = (
        COMFORT_WEIGHT_GAMES_PCT * result["games_played_pct"]
    ).clip(lower=0.0, upper=1.0)
    return result


def _ensure_columns(df: pd.DataFrame, columns: list[str]) -> None:
    """Raise ValueError when required columns are missing."""
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
