"""Opponent denial score adapter for ban prioritization."""

from __future__ import annotations

from typing import Dict

from core.scoring_draft_state import DraftState
from data.meta.load_u_gg_roles import load_roles
from ml.inference.counter_map import load_counter_map
from ml.inference.duos_map import load_duos_map
from ml.inference.role_filter import get_open_roles


def _games_to_pick_score(games_played: int) -> float:
    """Convert games played to a proportional pick score (same as comfort score calculation).
    
    Uses a linear scale from 0 to 15+ games:
    - 15+ games: 1.0
    - 0 games: 0.0
    - Linear interpolation in between
    """
    return min(1.0, max(0.0, games_played / 15.0))


def get_denial_scores(state: DraftState, opponent_profile: dict) -> Dict[str, float]:
    """Return denial scores for available champions.

    The opponent_profile is expected to include:
    - "frequent_picks": dict[champion, games_played] - raw games played count
    
    Pick score uses the same proportional scale as comfort score:
    - Linear scale from 0 to 15+ games (games_played / 15.0, clamped to [0, 1])
    
    This adapter is intended as a secondary modifier only.
    """
    duos_map = load_duos_map()
    counter_map = load_counter_map()
    roles_map = load_roles("data/meta/u_gg_roles.csv")
    
    if not roles_map:
        return {}
    
    # Get all champions from roles_map and opponent_profile
    all_champions = set(roles_map.keys())
    frequent = opponent_profile.get("frequent_picks", {})
    all_champions.update(frequent.keys())
    
    available = state.available_champions(all_champions)
    if not available:
        return {}

    opponent_picks = state.red_picks if state.acting_side() == "BLUE" else state.blue_picks
    our_picks = state.blue_picks if state.acting_side() == "BLUE" else state.red_picks
    
    # Determine opponent side for open roles calculation
    opponent_side = "RED" if state.acting_side() == "BLUE" else "BLUE"
    open_roles = get_open_roles(state, roles_map, opponent_side)

    scores: dict[str, float] = {}
    for champion in available:
        games_played = frequent.get(champion, 0)
        pick_score = _games_to_pick_score(int(games_played))
        synergy_score = _synergy_score(
            champion, opponent_picks, duos_map, roles_map, open_roles
        )
        counter_score = _counter_score(
            champion, our_picks, counter_map, roles_map, state
        )
        scores[champion] = (
            0.70 * synergy_score + 0.15 * pick_score + 0.15 * counter_score
        )

    return _clamp_scores(_normalize_scores(scores))


def _synergy_score(
    champion: str,
    opponent_picks: set[str],
    duos_map: dict[str, dict[str, dict[str, tuple[float, str, int]]]],
    roles_map: dict[str, set[str]],
    open_roles: set[str],
) -> float:
    """Compute synergy score with opponent's existing picks using U.GG duos data.
    
    Uses confidence weighting: score = base_score * min(1.0, matches / 250.0)
    where base_score = (winrate - 45) / 10, clamped to max 1.0.
    
    The duos CSV structure is: champion,role,duo_champion,duo_role,winrate,matches
    So we look up: opponent_pick -> opponent_pick_role -> candidate to find (winrate, duo_role, matches).
    We verify that the candidate can play the duo_role and that it's still open.
    """
    if not opponent_picks:
        return 0.0
    
    candidate_roles = roles_map.get(champion, set())
    if open_roles:
        candidate_roles = candidate_roles & open_roles
    if not candidate_roles:
        return 0.0

    CONFIDENCE_THRESHOLD = 250.0
    best_score = 0.0
    
    for pick in opponent_picks:
        pick_roles = roles_map.get(pick, set())
        if not pick_roles:
            continue
        
        # Check duos where the opponent's picked champion (in any of its roles) synergizes with the candidate
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


def _counter_score(
    champion: str,
    our_picks: set[str],
    counter_map: dict[str, dict[str, dict[str, tuple[float, int]]]],
    roles_map: dict[str, set[str]],
    state: DraftState,
) -> float:
    """Return the best counter score vs our current picks by role.
    
    Uses confidence weighting: score = base_score * min(1.0, matches / 250.0)
    where base_score = (winrate - 45) / 10, clamped to max 1.0.
    """
    if not our_picks:
        return 0.0
    if not roles_map:
        return 0.0
    opponent_side = "RED" if state.acting_side() == "BLUE" else "BLUE"
    open_roles = get_open_roles(state, roles_map, opponent_side)
    candidate_roles = roles_map.get(champion, set())
    if open_roles:
        candidate_roles = candidate_roles & open_roles
    if not candidate_roles:
        return 0.0
    
    CONFIDENCE_THRESHOLD = 250.0
    best_score = 0.0
    
    for pick in our_picks:
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




def _normalize_scores(scores: dict[str, float]) -> dict[str, float]:
    """Min-max normalize scores to [0, 1]."""
    if not scores:
        return {}
    values = list(scores.values())
    min_v = min(values)
    max_v = max(values)
    if min_v == max_v:
        return {champ: 0.0 for champ in scores}
    return {champ: (score - min_v) / (max_v - min_v) for champ, score in scores.items()}


def _clamp_scores(scores: dict[str, float]) -> dict[str, float]:
    """Clamp scores to [0, 1]."""
    return {champ: max(0.0, min(1.0, score)) for champ, score in scores.items()}
