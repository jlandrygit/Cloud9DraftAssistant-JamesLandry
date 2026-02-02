"""Roster map loading utilities."""

from __future__ import annotations

from functools import lru_cache
from os import getenv
from pathlib import Path

import json

DEFAULT_ROSTER_PATH = "data/rosters/roster_map.json"


@lru_cache(maxsize=1)
def load_roster_map(path: str | None = None) -> dict[str, dict[str, str]]:
    """Load a team -> role -> player roster map from disk."""
    roster_path = Path(path or getenv("ROSTER_MAP_PATH", DEFAULT_ROSTER_PATH))
    if not roster_path.exists():
        return {}
    try:
        with roster_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(team): dict(roles) for team, roles in data.items() if isinstance(roles, dict)}


def get_team_roster(team_name: str) -> dict[str, str]:
    """Return role -> player mapping for a given team name."""
    if not team_name:
        return {}
    roster_map = load_roster_map()
    return roster_map.get(team_name, {})
