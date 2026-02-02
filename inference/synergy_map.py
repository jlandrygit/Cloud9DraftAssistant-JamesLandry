"""Synergy map loading utilities."""

from __future__ import annotations

from functools import lru_cache
from os import getenv
from pathlib import Path

import json

DEFAULT_SYNERGY_PATH = "data/processed/synergy_map.json"


@lru_cache(maxsize=1)
def load_synergy_map(path: str | None = None) -> dict[str, dict[str, float]]:
    """Load a champion -> champion synergy map from disk."""
    target = Path(path or getenv("SYNERGY_MAP_PATH", DEFAULT_SYNERGY_PATH))
    if not target.exists():
        return {}
    try:
        with target.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    cleaned: dict[str, dict[str, float]] = {}
    for champ, inner in data.items():
        if not isinstance(inner, dict):
            continue
        cleaned[str(champ)] = {str(k): float(v) for k, v in inner.items()}
    return cleaned
