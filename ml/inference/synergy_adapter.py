"""Team synergy score adapter using u_gg_duos.csv."""

from __future__ import annotations

from typing import Dict

from core.scoring_draft_state import DraftState
from data.meta.load_u_gg_roles import load_roles
from ml.inference.duos_map import load_duos_map
from ml.inference.role_filter import get_open_roles


def get_synergy_scores(state: DraftState) -> Dict[str, float]:
    """Return synergy scores for available champions.

    - Pick phase: score candidates that synergize with our team's picks.
    - Ban phase: score candidates that synergize with opponent's picks.
    
    Uses confidence weighting: score = base_score * min(1.0, matches / 250.0)
    where base_score = (winrate - 45) / 10, clamped to max 1.0.
    """
    duos_map = load_duos_map()
    if not duos_map:
        return {}

    roles_map = load_roles("data/meta/u_gg_roles.csv")
    if not roles_map:
        return {}
    
    available = state.available_champions(set(roles_map.keys()))
    if not available:
        return {}

    # Determine which team's picks to check synergy with
    if state.is_pick_phase():
        # Pick phase: check synergy with our team's picks
        team_picks = (
            state.blue_picks if state.acting_side() == "BLUE" else state.red_picks
        )
        candidate_side = state.acting_side()
    else:
        # Ban phase: check synergy with opponent's picks
        team_picks = (
            state.red_picks if state.acting_side() == "BLUE" else state.blue_picks
        )
        candidate_side = "RED" if state.acting_side() == "BLUE" else "BLUE"
    
    open_roles = get_open_roles(state, roles_map, candidate_side)

    scores: dict[str, float] = {}
    for champion in available:
        scores[champion] = _max_synergy(
            champion, team_picks, duos_map, roles_map, open_roles
        )
    return scores


def _max_synergy(
    champion: str,
    team_picks: set[str],
    duos_map: dict[str, dict[str, dict[str, tuple[float, str, int]]]],
    roles_map: dict[str, set[str]],
    open_roles: set[str],
) -> float:
    """Return the best synergy score with current team picks by role.
    
    Uses confidence weighting based on matches: score = base_score * min(1.0, matches / 250.0)
    
    The duos CSV structure is: champion,role,duo_champion,duo_role,winrate,matches
    So we look up: pick -> pick_role -> candidate to find (winrate, duo_role, matches).
    We verify that the candidate can play the duo_role and that it's still open.
    """
    if not team_picks:
        return 0.0

    candidate_roles = roles_map.get(champion, set())
    if open_roles:
        candidate_roles = candidate_roles & open_roles
    if not candidate_roles:
        return 0.0

    CONFIDENCE_THRESHOLD = 250.0
    best_score = 0.0
    
    for pick in team_picks:
        pick_roles = roles_map.get(pick, set())
        if not pick_roles:
            continue
        
        # Check duos where the picked champion (in any of its roles) synergizes with the candidate
        # Look up: pick -> pick_role -> candidate -> (winrate, duo_role, matches)
        for pick_role in pick_roles:
            duo_data = duos_map.get(pick, {}).get(pick_role, {}).get(champion)
            if duo_data:
                winrate, duo_role, matches = duo_data
                # Verify the candidate can play the duo_role and it's still open
                if duo_role in candidate_roles and winrate > 0.0:
                    # Calculate base score
                    base_score = (winrate - 45.0) / 10.0
                    base_score = min(base_score, 1.0)  # Clamp base score to max 1.0
                    
                    # Apply confidence weighting
                    confidence = min(1.0, matches / CONFIDENCE_THRESHOLD)
                    score = base_score * confidence
                    
                    best_score = max(best_score, score)
    
    return best_score
