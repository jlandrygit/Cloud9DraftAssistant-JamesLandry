"""Duos map loading utilities from u_gg_duos.csv."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import pandas as pd

DEFAULT_DUOS_PATH = "data/meta/u_gg_duos.csv"


@lru_cache(maxsize=1)
def load_duos_map(path: str | None = None) -> dict[str, dict[str, dict[str, tuple[float, str, int]]]]:
    """Load a champion -> role -> duo_champion -> (winrate, duo_role, matches) map from u_gg_duos.csv.
    
    Returns a nested dict structure:
    {
        champion: {
            role: {
                duo_champion: (winrate, duo_role, matches)
            }
        }
    }
    """
    target = Path(path or DEFAULT_DUOS_PATH)
    if not target.exists():
        return {}
    
    try:
        df = pd.read_csv(target)
    except Exception:
        return {}
    
    if df.empty:
        return {}
    
    # Build nested dict: champion -> role -> duo_champion -> (winrate, duo_role, matches)
    duos_map: dict[str, dict[str, dict[str, tuple[float, str, int]]]] = {}
    
    for _, row in df.iterrows():
        champion = str(row["champion"]).strip()
        role = str(row["role"]).strip().upper()
        duo_champion = str(row["duo_champion"]).strip()
        duo_role = str(row["duo_role"]).strip().upper()
        winrate = float(row["winrate"])
        matches = int(float(row.get("matches", 0))) if pd.notna(row.get("matches")) else 0
        
        if champion not in duos_map:
            duos_map[champion] = {}
        if role not in duos_map[champion]:
            duos_map[champion][role] = {}
        duos_map[champion][role][duo_champion] = (winrate, duo_role, matches)
    
    return duos_map
