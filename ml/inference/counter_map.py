"""Counter map loading utilities."""

from __future__ import annotations

from functools import lru_cache
from os import getenv
from pathlib import Path
import csv

DEFAULT_COUNTER_PATH = "data/meta/u_gg_counters.csv"


@lru_cache(maxsize=1)
def load_counter_map(path: str | None = None) -> dict[str, dict[str, dict[str, tuple[float, int]]]]:
    """Load a champion -> role -> counter champion map from disk.
    
    Returns: champion -> role -> counter_champion -> (winrate, matches)
    """
    target = Path(path or getenv("COUNTER_MAP_PATH", DEFAULT_COUNTER_PATH))
    if not target.exists():
        return {}
    counters: dict[str, dict[str, dict[str, tuple[float, int]]]] = {}
    try:
        with target.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                champion = str(row.get("champion", "")).strip()
                role = str(row.get("role", "")).strip().upper()
                counter = str(row.get("counter_champion", "")).strip()
                winrate = row.get("winrate")
                matches_str = row.get("matches", "0")
                if not champion or not role or not counter:
                    continue
                try:
                    winrate_f = float(winrate)
                    matches_int = int(float(matches_str)) if matches_str and matches_str.strip() else 0
                except (TypeError, ValueError):
                    continue
                counters.setdefault(champion, {}).setdefault(role, {})[counter] = (winrate_f, matches_int)
    except Exception:
        return {}
    return counters
