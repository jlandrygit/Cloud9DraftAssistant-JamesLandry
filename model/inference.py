"""Lightweight inference helpers for demo recommendations."""

from __future__ import annotations

from typing import Any

from draft.draft_order import DRAFT_SEQUENCE
from draft.draft_state import DraftState
from inference.roster_map import get_team_roster
from inference.scoring_engine import DraftScoringEngine

_SCORING_ENGINE = DraftScoringEngine()


def recommend_picks(draft_state: dict) -> list[dict[str, Any]]:
    """Return pick recommendations for a draft state."""
    return _recommend(draft_state)


def recommend_bans(draft_state: dict) -> list[dict[str, Any]]:
    """Return ban recommendations for a draft state."""
    return _recommend(draft_state)


def _recommend(draft_state: dict) -> list[dict[str, Any]]:
    """Return recommendations using the shared scoring engine."""
    state = _payload_to_state(draft_state)
    roster = _payload_to_roster(draft_state)
    include_tier2 = draft_state.get("include_tier2", False)
    # Extract team names for ban-based comfort scoring
    # Determine which side is acting and extract appropriate team names
    acting_side = state.acting_side()
    if acting_side == "BLUE":
        # We're blue, opponent is red
        our_team_name = str(draft_state.get("blue_team") or draft_state.get("our_team") or "")
        opponent_team_name = str(draft_state.get("red_team") or draft_state.get("opponent_team") or "")
    else:
        # We're red, opponent is blue
        our_team_name = str(draft_state.get("red_team") or draft_state.get("opponent_team") or "")
        opponent_team_name = str(draft_state.get("blue_team") or draft_state.get("our_team") or "")
    recommendations = _SCORING_ENGINE.score(
        state, 
        opponent_roster=roster, 
        opponent_team_name=opponent_team_name if opponent_team_name else None,
        our_team_name=our_team_name if our_team_name else None,
        include_tier2=include_tier2
    )
    return [_to_payload(rec) for rec in recommendations[:5]]


def _payload_to_state(payload: dict[str, Any]) -> DraftState:
    """Convert UI/API payloads into the shared DraftState model."""
    blue_picks = set(payload.get("blue_picks") or [])
    red_picks = set(payload.get("red_picks") or [])
    blue_bans = set(payload.get("blue_bans") or [])
    red_bans = set(payload.get("red_bans") or [])
    fearless_bans = set(payload.get("fearless_bans") or [])
    # Include fearless_bans in the bans set so they're excluded from recommendations
    bans = blue_bans | red_bans | fearless_bans
    step_index = len(blue_picks) + len(red_picks) + len(blue_bans) + len(red_bans)
    if DRAFT_SEQUENCE:
        step_index = min(step_index, len(DRAFT_SEQUENCE) - 1)
    return DraftState(
        patch=str(payload.get("patch", payload.get("patch_version", "26.2"))),
        league=str(payload.get("league", "demo")),
        blue_team=str(payload.get("blue_team", "BLUE")),
        red_team=str(payload.get("red_team", "RED")),
        blue_picks=blue_picks,
        red_picks=red_picks,
        bans=bans,
        step_index=step_index,
    )


def _payload_to_roster(payload: dict[str, Any]) -> dict[str, dict[str, str]]:
    """Build a minimal roster map for comfort scoring."""
    blue_team = str(payload.get("blue_team") or payload.get("our_team") or "BLUE")
    red_team = str(payload.get("red_team") or payload.get("opponent_team") or "RED")
    blue_roster = get_team_roster(blue_team)
    red_roster = get_team_roster(red_team)
    return {
        "BLUE": blue_roster or {"UNKNOWN": blue_team},
        "RED": red_roster or {"UNKNOWN": red_team},
    }


def _to_payload(rec: Any) -> dict[str, Any]:
    """Map RecommendationExplanation into the UI payload format."""
    factors = ["policy model"]
    if rec.comfort_score > 0:
        factors.append("player comfort")
    if rec.counter_score > 0:
        factors.append("counter matchup")
    if rec.synergy_score > 0:
        factors.append("champion synergy")
    if rec.meta_score > 0:
        factors.extend(["meta win rate", "patch strength"])
    return {
        "champion": rec.champion,
        "score": float(rec.total_score),
        "explanation": rec.explanation_text,
        "policy_score": float(rec.policy_score),
        "comfort_score": float(rec.comfort_score),
        "counter_score": float(rec.counter_score),
        "synergy_score": float(rec.synergy_score),
        "meta_score": float(rec.meta_score),
        "contributing_factors": factors,
    }
