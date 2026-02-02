"""Meta strength score adapter using pre-calculated scores."""

from __future__ import annotations

from functools import lru_cache
from os import getenv
from pathlib import Path
from typing import Dict

import joblib
import pandas as pd

from data.meta.load_u_gg_roles import load_roles
from draft.draft_state import DraftState
from inference.role_filter import get_open_roles

DEFAULT_META_SCORES_PATH = "data/meta/meta_scores.csv"


@lru_cache(maxsize=1)
def _load_meta_scores(path: str | None = None) -> dict[str, dict[str, float]]:
    """Load pre-calculated meta scores from CSV.
    
    Returns:
        dict[champion, dict[role, meta_score]]
    """
    target = Path(path or getenv("META_SCORES_PATH", DEFAULT_META_SCORES_PATH))
    if not target.exists():
        return {}
    
    try:
        df = pd.read_csv(target)
    except Exception:
        return {}
    
    if df.empty:
        return {}
    
    required = {"champion", "role", "meta_score"}
    if not required.issubset(df.columns):
        return {}
    
    scores: dict[str, dict[str, float]] = {}
    for _, row in df.iterrows():
        champion = str(row["champion"]).strip()
        role = str(row["role"]).strip().upper()
        meta_score = float(row["meta_score"])
        
        if champion not in scores:
            scores[champion] = {}
        scores[champion][role] = meta_score
    
    return scores


def get_meta_scores(state: DraftState) -> Dict[str, float]:
    """Return meta strength scores for available champions.
    
    Uses pre-calculated meta scores from meta_scores.csv.
    For each champion, selects the best score from available open roles.
    """
    meta_scores_by_role = _load_meta_scores()
    if not meta_scores_by_role:
        return {}
    
    # Get all champions that have meta scores - this is our source of truth
    champions_with_scores = set(meta_scores_by_role.keys())
    # Get champion vocab if available
    champion_vocab = _load_champion_vocab()
    # Combine all possible champions to ensure we don't miss any
    all_champions = set(champion_vocab) if champion_vocab else set()
    all_champions = all_champions | champions_with_scores
    
    # Get available champions (not picked/banned)
    available = state.available_champions(all_champions)
    
    roles_map = _load_roles()
    acting = state.acting_side()
    target_side = acting if state.is_pick_phase() else ("RED" if acting == "BLUE" else "BLUE")
    open_roles = get_open_roles(state, roles_map, target_side)
    
    scores: dict[str, float] = {}
    # Process ALL champions that have meta scores
    # This ensures we calculate scores for all champions, regardless of whether they're in the vocab
    # The scoring engine will filter to only use scores for available champions
    for champion in champions_with_scores:
        champion_scores = meta_scores_by_role.get(champion, {})
        if not champion_scores:
            # Only include 0.0 scores for available champions to avoid cluttering the dict
            if champion in available:
                scores[champion] = 0.0
            continue
        
        # Get all roles this champion can play from roles_map
        champ_roles_from_map = roles_map.get(champion, set())
        # Get all roles that have meta scores available (this is the source of truth for available roles)
        champ_roles_from_stats = set(champion_scores.keys())
        
        # Combine both sources to get all possible roles
        # This ensures we don't miss any roles
        all_champ_roles = champ_roles_from_map | champ_roles_from_stats
        
        # Filter to only roles that are still open (not filled) for this team
        # This is the primary filter - we want roles that are both:
        # 1. Roles the champion can play (from all_champ_roles)
        # 2. Roles that are still open for this team (from open_roles)
        eligible_roles = {role for role in all_champ_roles if role in open_roles}
        
        # If no eligible roles (all filled or no overlap with open_roles), 
        # fall back to all roles that have scores available
        # This ensures we always calculate a score if scores are available
        if not eligible_roles:
            eligible_roles = champ_roles_from_stats  # Use stats keys to ensure we have scores
        
        # Find the best score across all eligible roles
        # Loop through each eligible role and get the score for that role
        best = None
        for role in eligible_roles:
            # Get the pre-calculated meta score for this champion in this role
            # champion_scores is a dict[role, meta_score] for this champion
            score = champion_scores.get(role)
            if score is not None:
                # Track the best (highest) score across all eligible roles
                if best is None:
                    best = score
                else:
                    best = max(best, score)
        
        # Final fallback: if we still don't have a score but have champion_scores,
        # use the best score from any available role (this should always work if champion_scores exists)
        if best is None:
            if champion_scores:
                # Use the maximum score from any role - this ensures we always get a score
                best = max(champion_scores.values())
            else:
                best = 0.0
        
        scores[champion] = best
    
    return scores


@lru_cache(maxsize=1)
def _load_champion_vocab() -> list[str]:
    """Load champion vocabulary from the processed encoder."""
    try:
        encoder = joblib.load("data/processed/champion_encoder.pkl")
        return list(encoder.classes_)
    except Exception:
        return []


@lru_cache(maxsize=1)
def _load_roles() -> dict[str, set[str]]:
    """Load champion role mappings."""
    path = getenv("UGG_ROLES_PATH", "data/meta/u_gg_roles.csv")
    try:
        return load_roles(path)
    except Exception:
        return {}
