"""Rule-based ban recommendation engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol

from core.draft_state import DraftState, Side


@dataclass(frozen=True)
class BanReasoning:
    """Explain why a champion was recommended as a ban."""

    opponent_comfort: float
    global_strength: float
    synergy_denial: float


@dataclass(frozen=True)
class BanRecommendation:
    """Ban recommendation with reasoning metadata."""

    champion_id: str
    score: float
    reasoning: BanReasoning


class OpponentComfortProvider(Protocol):
    """Provides opponent comfort scores for champions."""

    def score(self, opponent_id: str, champion_id: str) -> float:
        """Return opponent comfort score for a champion."""
        ...


class GlobalStrengthProvider(Protocol):
    """Provides global champion strength by patch."""

    def score(self, champion_id: str, patch: str) -> float:
        """Return champion strength for a specific patch."""
        ...


class SynergyProvider(Protocol):
    """Provides team synergy scores for champion pairs."""

    def score(self, champion_id: str, teammate_id: str) -> float:
        """Return synergy score between two champions."""
        ...


class BanRecommendationEngine:
    """Rule-based engine for top ban recommendations.

    The scoring heuristic combines:
    - Opponent comfort (how strong the opponent is on the champ)
    - Global strength (patch-aware meta power)
    - Synergy denial (preventing strong combos with enemy picks)
    """

    def __init__(
        self,
        *,
        comfort_provider: OpponentComfortProvider,
        strength_provider: GlobalStrengthProvider,
        synergy_provider: SynergyProvider,
        candidate_pool: Iterable[str],
    ) -> None:
        self._comfort_provider = comfort_provider
        self._strength_provider = strength_provider
        self._synergy_provider = synergy_provider
        self._candidate_pool = list(candidate_pool)

    def recommend_bans(
        self,
        draft_state: DraftState,
        opponent_id: str,
        patch: str,
        for_side: Side,
        limit: int = 3,
    ) -> list[BanRecommendation]:
        """Return top ban recommendations with reasoning metadata."""
        enemy_picks = self._get_enemy_picks(draft_state, for_side)
        recommendations: list[BanRecommendation] = []

        for champion_id in self._candidate_pool:
            inputs = BanScoreInputs(
                champion_id=champion_id,
                enemy_picks=enemy_picks,
                opponent_comfort=self._comfort_provider.score(opponent_id, champion_id),
                global_strength=self._strength_provider.score(champion_id, patch),
                synergy_denial=self._deny_synergy(champion_id, enemy_picks),
            )
            score = self._score(inputs)

            recommendations.append(
                BanRecommendation(
                    champion_id=champion_id,
                    score=score,
                    reasoning=BanReasoning(
                        opponent_comfort=inputs.opponent_comfort,
                        global_strength=inputs.global_strength,
                        synergy_denial=inputs.synergy_denial,
                    ),
                )
            )

        recommendations.sort(key=lambda rec: rec.score, reverse=True)
        return recommendations[:limit]

    def _get_enemy_picks(self, state: DraftState, for_side: Side) -> list[str]:
        """Collect enemy picks for synergy denial scoring."""
        # Enemy picks are always the opposite side from the evaluation target.
        picks = (
            state.red_picks.to_dict()
            if for_side is Side.BLUE
            else state.blue_picks.to_dict()
        )
        return [champion for champion in picks.values() if champion]

    def _deny_synergy(self, candidate_id: str, enemy_picks: list[str]) -> float:
        """Return the max synergy score vs enemy picks.

        This uses a max-heuristic (not an average/aggregate) to focus on the
        strongest potential combo we want to deny in the current enemy draft.
        """
        if not enemy_picks:
            return 0.0
        # Heuristic: block the strongest potential combo with current enemy picks.
        return max(self._synergy_provider.score(candidate_id, pick) for pick in enemy_picks)

    def _score(self, inputs: "BanScoreInputs") -> float:
        """Compute a weighted score from validated inputs."""
        # Weighting prioritizes opponent comfort, then global strength, then synergy denial.
        weights = (0.45, 0.35, 0.20)
        return (
            weights[0] * inputs.opponent_comfort
            + weights[1] * inputs.global_strength
            + weights[2] * inputs.synergy_denial
        )


@dataclass(frozen=True)
class BanScoreInputs:
    """Centralized inputs for ban scoring with basic validation."""

    champion_id: str
    enemy_picks: list[str]
    opponent_comfort: float
    global_strength: float
    synergy_denial: float

    def __post_init__(self) -> None:
        if not isinstance(self.enemy_picks, list):
            raise TypeError("enemy_picks must be a list of champion IDs.")
        for pick in self.enemy_picks:
            if not isinstance(pick, str):
                raise TypeError("enemy_picks must contain only strings.")
        for name, value in (
            ("opponent_comfort", self.opponent_comfort),
            ("global_strength", self.global_strength),
            ("synergy_denial", self.synergy_denial),
        ):
            if not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a numeric score.")
