"""Opponent profile builder from historical data."""

from __future__ import annotations

from functools import lru_cache
from os import getenv
from pathlib import Path

import pandas as pd

from ml.inference.synergy_map import load_synergy_map

DEFAULT_STATS_PATH = "data/processed/player_vs_opponent_champion_stats.csv"


@lru_cache(maxsize=64)
def build_opponent_profile(opponent_team: str) -> dict[str, dict[str, float]]:
    """Build a minimal opponent profile from aggregated historical stats.
    
    Returns:
        - "frequent_picks": dict[champion, games_played] - raw games played count
        - "high_win_comps": dict[champion, win_rate] - normalized win rate
        - "synergy_map": dict - historical synergy map (legacy, not used)
    """
    if not opponent_team:
        return {}
    df = _load_stats()
    if df.empty:
        return {}
    matched_team = _match_opponent_team(opponent_team, df["opponent_team_name"].unique())
    if not matched_team:
        return {}
    subset = df[df["opponent_team_name"] == matched_team]
    if subset.empty:
        return {}

    games_by_champ = subset.groupby("champion_name")["games_played"].sum()
    # Store raw games_played counts (same calculation as comfort score)
    frequent_picks = games_by_champ.fillna(0).astype(int).to_dict()

    winrate_by_champ = subset.groupby("champion_name")["win_rate"].mean()
    high_win_comps = (
        winrate_by_champ.fillna(0.0).clip(0.0, 1.0).to_dict()
        if not winrate_by_champ.empty
        else {}
    )

    synergy_map = load_synergy_map()

    return {
        "frequent_picks": {str(k): int(v) for k, v in frequent_picks.items()},  # Store raw games_played
        "high_win_comps": {str(k): float(v) for k, v in high_win_comps.items()},
        "synergy_map": synergy_map,
    }


@lru_cache(maxsize=1)
def _load_stats() -> pd.DataFrame:
    """Load aggregated opponent stats from disk."""
    path = Path(getenv("OPPONENT_STATS_PATH", DEFAULT_STATS_PATH))
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    required = {
        "opponent_team_name",
        "champion_name",
        "games_played",
        "win_rate",
    }
    if not required.issubset(df.columns):
        return pd.DataFrame()
    return df


def _match_opponent_team(opponent_team: str, candidates: list[str]) -> str | None:
    """Best-effort match between short codes and dataset team names."""
    def normalize(value: str) -> str:
        return "".join(ch for ch in value.upper() if ch.isalnum())

    query = normalize(opponent_team)
    if not query:
        return None
    normalized = {name: normalize(name) for name in candidates}
    for name, norm in normalized.items():
        if norm == query:
            return name
    contains = [name for name, norm in normalized.items() if query in norm or norm in query]
    if not contains:
        return None
    return max(contains, key=lambda name: len(normalized[name]))
