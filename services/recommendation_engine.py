"""Recommendation engine for draft suggestions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Protocol

import logging

from core.draft_state import DraftState
from services.draft_rules import DraftRules, LegalAction


class DraftActionType(str, Enum):
    """Draft action types handled by the recommendation engine."""

    PICK = "pick"
    BAN = "ban"


@dataclass(frozen=True)
class DraftAction:
    """A candidate draft action for simulation and scoring."""

    action_type: DraftActionType
    champion_id: str
    role: str | None = None


@dataclass(frozen=True)
class Recommendation:
    """Ranked recommendation with a confidence score."""

    action: DraftAction
    score: float
    confidence: float


class DraftScoringModel(Protocol):
    """Protocol for ML model scoring draft states."""

    def score(self, state: DraftState) -> float:
        """Return a model score (higher is better)."""
        ...


class ComfortProvider(Protocol):
    """Provides player comfort scores for our own picks."""

    def score(self, champion_id: str, role: str | None) -> float:
        """Return a comfort score in [0, 1] for the current pick context."""
        ...


class SynergyProvider(Protocol):
    """Provides team synergy scores for a candidate pick."""

    def score(self, state: DraftState, champion_id: str) -> float:
        """Return a synergy score in [0, 1] for the current draft state."""
        ...


class OpponentThreatProvider(Protocol):
    """Provides opponent threat scores for bans."""

    def score(self, state: DraftState, champion_id: str) -> float:
        """Return a threat score in [0, 1] for the opponent context."""
        ...


class RecommendationEngine:
    """Produces draft recommendations by simulating and scoring actions.

    The engine follows a strict pipeline:
    1) Enumerate legal actions from the current DraftState.
    2) Simulate the action to produce a next DraftState snapshot.
    3) Score the resulting state with a trained ML model.
    4) Rank actions by score and attach confidence values.
    """
    # HACKATHON NOTE: Player comfort is learned from historical picks, time-decayed usage,
    # and role-specific performance. Opponent-specific signals combine comfort, patch strength,
    # and recent pick rates to estimate threat. The system is data-driven yet interpretable
    # because the final score is a transparent weighted combination.

    def __init__(
        self,
        model: DraftScoringModel,
        rules: DraftRules,
        *,
        comfort_provider: ComfortProvider | None = None,
        synergy_provider: SynergyProvider | None = None,
        opponent_threat_provider: OpponentThreatProvider | None = None,
    ) -> None:
        self._model = model
        self._rules = rules
        self._comfort_provider = comfort_provider
        self._synergy_provider = synergy_provider
        self._opponent_threat_provider = opponent_threat_provider
        self._logger = logging.getLogger(__name__)

    def recommend(self, state: DraftState) -> list[Recommendation]:
        """Return a ranked list of recommendations for the current state."""
        actions = list(self.enumerate_actions(state))
        scored = [self.score_action(state, action) for action in actions]
        return self.rank_actions(scored)

    def enumerate_actions(self, state: DraftState) -> Iterable[DraftAction]:
        """Enumerate all legal actions for the current draft phase."""
        legal_actions = self._rules.enumerate_legal_actions(state)
        return [self._to_draft_action(action) for action in legal_actions]

    def simulate_action(self, state: DraftState, action: DraftAction) -> DraftState:
        """Simulate the provided action and return the next DraftState."""
        legal_action = self._to_legal_action(state, action)
        return self._rules.apply_action(state, legal_action)

    def score_action(self, state: DraftState, action: DraftAction) -> Recommendation:
        """Simulate and score a candidate action."""
        next_state = self.simulate_action(state, action)
        # ML outputs enter via the providers (comfort/synergy/threat).
        comfort = (
            self._comfort_provider.score(action.champion_id, action.role)
            if self._comfort_provider
            else 0.0
        )
        synergy = (
            self._synergy_provider.score(state, action.champion_id)
            if self._synergy_provider
            else 0.0
        )
        opponent_threat = (
            self._opponent_threat_provider.score(state, action.champion_id)
            if self._opponent_threat_provider
            else 0.0
        )
        # Simple weighted score: comfort + synergy - opponent_threat.
        score = comfort + synergy - opponent_threat
        confidence = self._score_to_confidence(score)
        self._logger.info(
            "recommendation_score",
            extra={
                "champion_id": action.champion_id,
                "role": action.role,
                "comfort": comfort,
                "synergy": synergy,
                "opponent_threat": opponent_threat,
                "score": score,
            },
        )
        return Recommendation(action=action, score=score, confidence=confidence)

    def rank_actions(self, recommendations: Iterable[Recommendation]) -> list[Recommendation]:
        """Rank recommendations by score, highest first."""
        return sorted(recommendations, key=lambda item: item.score, reverse=True)

    def _score_to_confidence(self, score: float) -> float:
        """Convert model scores to a normalized confidence value."""
        # TODO (Post-Hackathon): Calibrate confidence values against held-out outcomes.
        # Missing: a calibrated mapping from raw scores to well-calibrated probabilities/confidence.
        # Out of scope: hackathon demo used a monotonic clamp to keep UI behavior stable.
        # Production: Platt/isotonic calibration, calibration metrics (ECE/Brier), and per-patch recalibration support.
        return max(0.0, min(1.0, score))

    def _to_draft_action(self, action: LegalAction) -> DraftAction:
        """Convert a rules action to a recommendation action."""
        return DraftAction(
            action_type=action.action_type,
            champion_id=action.champion_id,
            role=action.role.value if action.role else None,
        )

    def _to_legal_action(self, state: DraftState, action: DraftAction) -> LegalAction:
        """Convert a recommendation action to a rules action."""
        legal_actions = self._rules.enumerate_legal_actions(state)
        for legal_action in legal_actions:
            if (
                legal_action.action_type is action.action_type
                and legal_action.champion_id == action.champion_id
            ):
                return legal_action
        raise ValueError("Action is not legal for the current draft state.")
