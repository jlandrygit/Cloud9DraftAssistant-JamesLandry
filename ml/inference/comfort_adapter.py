"""Comfort score adapter for draft-time ranking."""

from __future__ import annotations

from functools import lru_cache
from os import getenv
from pathlib import Path
from typing import Dict, Tuple

import joblib
import pandas as pd

from data.meta.load_u_gg_roles import load_roles
from core.scoring_draft_state import DraftState
from ml.inference.role_filter import get_open_roles
from ml.player_comfort_model import FEATURES_PATH

STATS_PATH = Path("data/processed/player_champion_stats.csv")
BAN_STATS_PATH = Path("data/processed/team_ban_stats.csv")

@lru_cache(maxsize=1)
def _load_champion_vocab() -> list[str]:
    """Load champion vocabulary from the processed encoder."""
    try:
        encoder = joblib.load("data/processed/champion_encoder.pkl")
        return list(encoder.classes_)
    except Exception:
        return _load_champion_vocab_from_stats()


def _load_champion_vocab_from_stats() -> list[str]:
    """Fallback champion vocabulary from player comfort stats."""
    try:
        if FEATURES_PATH.suffix == ".parquet":
            df = pd.read_parquet(FEATURES_PATH)
        else:
            df = pd.read_csv(FEATURES_PATH)
    except Exception:
        return []
    if "champion_name" not in df.columns:
        return []
    return sorted({str(name) for name in df["champion_name"].dropna().unique()})


@lru_cache(maxsize=1)
def _load_role_map() -> dict[str, set[str]]:
    """Load champion -> roles mapping from U.GG roles data."""
    path = getenv("UGG_ROLES_PATH", "data/meta/u_gg_roles.csv")
    try:
        return load_roles(path)
    except Exception:
        return {}


@lru_cache(maxsize=1)
def _load_player_champion_stats() -> dict[tuple[str, str, str], Tuple[int, float]]:
    """Load games played and win rate keyed by (player, role, champion).
    
    Returns:
        dict mapping (player, role, champion) -> (games_played, win_rate)
    """
    try:
        df = pd.read_csv(STATS_PATH)
    except Exception:
        return {}
    required = {"player_name", "role", "champion_name", "games_played", "win_rate"}
    if not required.issubset(df.columns):
        return {}
    mapping: dict[tuple[str, str, str], Tuple[int, float]] = {}
    for _, row in df[list(required)].dropna().iterrows():
        player = str(row["player_name"]).strip()
        role = str(row["role"]).strip().upper()
        champion = str(row["champion_name"]).strip()
        if not player or not role or not champion:
            continue
        try:
            games = int(row["games_played"])
            win_rate = float(row["win_rate"])
            # Clamp win_rate to [0, 1]
            win_rate = max(0.0, min(1.0, win_rate))
        except Exception:
            continue
        mapping[(player, role, champion)] = (games, win_rate)
    return mapping


@lru_cache(maxsize=1)
def _load_team_ban_stats() -> dict[tuple[str, str], float]:
    """Load team ban statistics keyed by (team_name, champion_name).
    
    Returns:
        dict mapping (team_name, champion_name) -> ban_rate
    """
    try:
        if not BAN_STATS_PATH.exists():
            return {}
        df = pd.read_csv(BAN_STATS_PATH)
    except Exception:
        return {}
    required = {"team_name", "champion_name", "ban_rate"}
    if not required.issubset(df.columns):
        return {}
    mapping: dict[tuple[str, str], float] = {}
    for _, row in df[list(required)].dropna().iterrows():
        team = str(row["team_name"]).strip()
        champion = str(row["champion_name"]).strip()
        if not team or not champion:
            continue
        try:
            ban_rate = float(row["ban_rate"])
            # Clamp ban_rate to [0, 1]
            ban_rate = max(0.0, min(1.0, ban_rate))
        except Exception:
            continue
        mapping[(team, champion)] = ban_rate
    return mapping


def _games_to_comfort(games_played: int) -> float:
    """Convert games played to a comfort value.
    
    Uses a linear scale from 0 to 20+ games:
    - 20+ games: 1.0 (from games alone)
    - 0 games: 0.0
    - Linear interpolation in between
    """
    return games_played / 20.0


def get_comfort_scores(
    state: DraftState, 
    opponent_roster: dict,
    opponent_team_name: str | None = None,
    our_team_name: str | None = None
) -> Dict[str, float]:
    """Return comfort scores for available champions.

    Assumptions:
    - opponent_roster is a dict keyed by side ("BLUE"/"RED") with role->player mappings.
      Example: {"BLUE": {"TOP": "PlayerA", ...}, "RED": {"TOP": "PlayerB", ...}}
    - If roles are missing, the adapter falls back to "UNKNOWN".
    - opponent_team_name is the name of the opponent team (for ban-based comfort).
    - our_team_name is the name of our team (for pick-based comfort).

    Behavior:
    - Pick phase: score champions high if:
      1. The acting side's players have high comfort (games played)
      2. The champion is frequently banned against OUR team (strong pick for us)
    - Ban phase: score champions high if:
      1. The opposing side's players have high comfort (deny their comfort)
      2. The champion is frequently banned against OPPONENT team (we should ban it)
    This adapter is intended as a secondary modifier only.
    """
    champions = _load_champion_vocab()
    if not champions:
        return {}

    available = state.available_champions(set(champions))
    if not available:
        return {}

    acting = state.acting_side()
    target_side = acting if state.is_pick_phase() else ("RED" if acting == "BLUE" else "BLUE")
    roster = opponent_roster.get(target_side, {})
    if not isinstance(roster, dict) or not roster:
        roster = {"UNKNOWN": "UNKNOWN"}

    role_map = _load_role_map()
    stats_map = _load_player_champion_stats()
    ban_stats_map = _load_team_ban_stats()
    open_roles = get_open_roles(state, role_map, target_side)
    scores: dict[str, float] = {}
    
    for champion in available:
        roles = role_map.get(champion, set())
        if not roles:
            continue
        candidate_roles = roles & open_roles if open_roles else roles
        if not candidate_roles:
            continue
        
        # Calculate player-based comfort
        best: float | None = None
        for role in candidate_roles:
            player = roster.get(role)
            if not player or not str(player).strip():
                continue
            stats = stats_map.get((str(player).strip(), str(role).upper(), champion))
            if stats:
                games, _ = stats  # Ignore win_rate
                score = _games_to_comfort(games)
                best = score if best is None else max(best, score)
        
        # Start with player-based comfort (or 0.0 if no player data)
        player_comfort = best if best is not None else 0.0
        
        # Add ban-based comfort bonus based on phase
        ban_bonus = 0.0
        if state.is_pick_phase():
            # Pick phase: Use OUR team's ban stats
            # If a champion is frequently banned against us, it's a strong pick for us
            if our_team_name:
                ban_rate = ban_stats_map.get((our_team_name, champion), 0.0)
                ban_bonus = ban_rate
        else:
            # Ban phase: Use OPPONENT team's ban stats
            # If a champion is frequently banned against the opponent, we should ban it
            if opponent_team_name:
                ban_rate = ban_stats_map.get((opponent_team_name, champion), 0.0)
                ban_bonus = ban_rate
        
        # Combine player comfort and ban bonus, then clamp to [0, 1]
        total_comfort = min(1.0, max(0.0, player_comfort + ban_bonus))
        
        if total_comfort > 0.0:
            scores[champion] = total_comfort

    return _clamp_scores(scores)


def _clamp_scores(scores: dict[str, float]) -> dict[str, float]:
    """Clamp scores to [0, 1]."""
    return {champ: max(0.0, min(1.0, score)) for champ, score in scores.items()}
