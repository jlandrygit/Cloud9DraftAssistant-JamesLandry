"""Shared explanation objects for UI and API."""

from __future__ import annotations

from dataclasses import dataclass, asdict

from draft.draft_state import DraftState


@dataclass(frozen=True)
class RecommendationExplanation:
    """Single source of truth for recommendation explanations.

    This object should be produced by inference and passed directly to UI/API
    layers to avoid drift between scores and human-facing explanations.
    """

    champion: str
    total_score: float
    policy_score: float
    comfort_score: float
    counter_score: float
    synergy_score: float
    meta_score: float
    policy_contribution_pct: float
    modifier_contribution_pct: float
    explanation_text: str

    def to_dict(self) -> dict[str, float | str]:
        """Serialize the explanation to a JSON-friendly dict."""
        return asdict(self)

    def __post_init__(self) -> None:
        if self.policy_contribution_pct < 70.0:
            raise ValueError("policy_contribution_pct must be >= 70")
        total = self.policy_contribution_pct + self.modifier_contribution_pct
        if abs(total - 100.0) > 1e-6:
            raise ValueError("Contribution percentages must sum to 100")

    @staticmethod
    def build_explanation(
        champion: str, scores_dict: dict[str, float], state: DraftState
    ) -> str:
        """Build a concise explanation based on top contributing scores."""
        phase = "pick" if state.is_pick_phase() else "ban"
        ordered = sorted(scores_dict.items(), key=lambda item: item[1], reverse=True)
        top_labels = [name for name, score in ordered if score > 0 and name != "policy"][:2]

        base = (
            f"Recommended primarily based on professional draft patterns for this {phase}."
        )
        if not top_labels:
            return base

        phrased = []
        for label in top_labels:
            if label == "meta":
                phrased.append("currently strong on this patch")
            else:
                phrased.append(label)

        reinforcing = ", ".join(phrased)
        return f"{base} Reinforced by {reinforcing}."
