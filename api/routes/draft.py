"""Draft-related HTTP endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from core.draft_state import (
    BanPhase,
    BanPhaseBans,
    DraftPhase,
    DraftState,
    Role,
    RolePicks,
    Side,
)
from inference.scoring_engine import DraftScoringEngine
from storage.draft_state_store import InMemoryDraftStateStore

router: APIRouter = APIRouter()

# TODO (Post-Hackathon): Replace the in-memory draft session store with a Redis-backed DraftStateStore.
# Missing: shared, durable session state across processes/instances (required for multi-worker FastAPI).
# Out of scope: hackathon demo assumed a single-process deployment and no persistence requirements.
# Production: add Redis store with TTL, serialization/versioning, concurrency-safe updates, and integration tests.
_DRAFT_STORE = InMemoryDraftStateStore()

_SCORING_ENGINE = DraftScoringEngine()


class DraftUpdateRequest(BaseModel):
    """Payload for updating a live draft session."""

    session_id: str = Field(..., min_length=1)
    blue_team: str = Field(..., min_length=1)
    red_team: str = Field(..., min_length=1)
    phase: DraftPhase
    turn: int = Field(..., ge=0)
    blue_picks: dict[Role, str | None] = Field(default_factory=dict)
    red_picks: dict[Role, str | None] = Field(default_factory=dict)
    ban_phases: dict[BanPhase, dict[Side, list[str]]] = Field(default_factory=dict)


class DraftUpdateResponse(BaseModel):
    """Response wrapper for updated draft state."""

    session_id: str
    state: dict[str, Any]


class RecommendationsResponse(BaseModel):
    """Draft recommendation response."""

    session_id: str
    recommendations: list[dict[str, Any]]


class BanRecommendationsResponse(BaseModel):
    """Ban recommendation response."""

    session_id: str
    bans: list[dict[str, Any]]


@router.post("/update", response_model=DraftUpdateResponse)
def update_draft(request: DraftUpdateRequest) -> DraftUpdateResponse:
    """Update or initialize an in-memory draft session."""
    state = DraftState(
        blue_team=request.blue_team,
        red_team=request.red_team,
        phase=request.phase,
        turn=request.turn,
        blue_picks=_build_role_picks(request.blue_picks),
        red_picks=_build_role_picks(request.red_picks),
        ban_phases=_build_ban_phases(request.ban_phases),
    )
    _DRAFT_STORE.set(request.session_id, state)
    return DraftUpdateResponse(session_id=request.session_id, state=state.to_dict())


@router.get("/recommendations", response_model=RecommendationsResponse)
def get_recommendations(session_id: str) -> RecommendationsResponse:
    """Return draft recommendations for a session."""
    state = _get_state_or_404(session_id)
    recommendations = [
        rec.to_dict() for rec in _SCORING_ENGINE.score(state)[:5]
    ]
    return RecommendationsResponse(session_id=session_id, recommendations=recommendations)


@router.get("/bans", response_model=BanRecommendationsResponse)
def get_ban_recommendations(session_id: str) -> BanRecommendationsResponse:
    """Return ban recommendations for a session."""
    state = _get_state_or_404(session_id)
    bans = [
        rec.to_dict() for rec in _SCORING_ENGINE.score(state)[:5]
    ]
    return BanRecommendationsResponse(session_id=session_id, bans=bans)


def _get_state_or_404(session_id: str) -> DraftState:
    """Retrieve a draft session or raise an HTTP 404."""
    state = _DRAFT_STORE.get(session_id)
    if not state:
        raise HTTPException(status_code=404, detail="Draft session not found.")
    return state


def get_active_session_count() -> int:
    """Return the number of active draft sessions."""
    return _DRAFT_STORE.count()


def get_session_ttl_seconds() -> int:
    """Return the configured draft session TTL."""
    return _DRAFT_STORE.ttl_seconds()


def _build_role_picks(raw: dict[Role, str | None]) -> RolePicks:
    """Convert role pick maps into RolePicks dataclass."""
    return RolePicks(
        top=raw.get(Role.TOP),
        jungle=raw.get(Role.JUNGLE),
        mid=raw.get(Role.MID),
        bottom=raw.get(Role.BOTTOM),
        support=raw.get(Role.SUPPORT),
    )


def _build_ban_phases(
    raw: dict[BanPhase, dict[Side, list[str]]]
) -> tuple[BanPhaseBans, ...]:
    """Convert ban phase payloads into BanPhaseBans tuples."""
    return tuple(
        BanPhaseBans(
            phase=phase,
            blue_bans=tuple(sides.get(Side.BLUE, [])),
            red_bans=tuple(sides.get(Side.RED, [])),
        )
        for phase, sides in raw.items()
    )


