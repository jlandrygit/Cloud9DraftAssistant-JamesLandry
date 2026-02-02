"""Counter score adapter for pick/ban recommendations."""

from __future__ import annotations

from typing import Dict

from draft.draft_state import DraftState
from data.meta.load_u_gg_roles import load_roles
from inference.counter_map import load_counter_map
from inference.role_filter import get_open_roles


def get_counter_scores(state: DraftState) -> Dict[str, float]:
    """Return counter scores for available champions.

    - Pick phase: score candidates that counter enemy picks.
    - Ban phase: score candidates that counter our current picks.
    
    Uses confidence weighting: score = base_score * min(1.0, matches / 250.0)
    where base_score = (winrate - 45) / 10, clamped to max 1.0.
    """
    counter_map = load_counter_map()
    if not counter_map:
        return {}

    roles_map = load_roles("data/meta/u_gg_roles.csv")
    if not roles_map:
        return {}
    available = state.available_champions(set(roles_map.keys()))
    if not available:
        return {}

    if state.is_pick_phase():
        enemy_picks = state.red_picks if state.acting_side() == "BLUE" else state.blue_picks
    else:
        enemy_picks = state.blue_picks if state.acting_side() == "BLUE" else state.red_picks

    candidate_side = state.acting_side() if state.is_pick_phase() else (
        "RED" if state.acting_side() == "BLUE" else "BLUE"
    )
    open_roles = get_open_roles(state, roles_map, candidate_side)

    scores: dict[str, float] = {}
    for champion in available:
        scores[champion] = _max_counter(
            champion, enemy_picks, counter_map, roles_map, open_roles
        )
    return scores


def _max_counter(
    champion: str,
    enemy_picks: set[str],
    counter_map: dict[str, dict[str, dict[str, tuple[float, int]]]],
    roles_map: dict[str, set[str]],
    open_roles: set[str],
) -> float:
    """Return the best counter score vs current enemy picks by role.
    
    Uses confidence weighting based on matches: score = base_score * min(1.0, matches / 250.0)
    """
    if not enemy_picks:
        return 0.0

    candidate_roles = roles_map.get(champion, set())
    if open_roles:
        candidate_roles = candidate_roles & open_roles
    if not candidate_roles:
        return 0.0

    CONFIDENCE_THRESHOLD = 250.0
    best_score = 0.0
    
    for pick in enemy_picks:
        target_roles = roles_map.get(pick, set())
        if not target_roles:
            continue
        for role in candidate_roles & target_roles:
            counter_data = counter_map.get(pick, {}).get(role, {}).get(champion)
            if counter_data:
                winrate, matches = counter_data
                # Calculate base score
                base_score = (winrate - 45.0) / 10.0
                base_score = min(base_score, 1.0)  # Clamp base score to max 1.0
                
                # Apply confidence weighting
                confidence = min(1.0, matches / CONFIDENCE_THRESHOLD)
                score = base_score * confidence
                
                best_score = max(best_score, score)
    
    return best_score
