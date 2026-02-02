"""Pydantic schemas for recommendation explanations."""

from __future__ import annotations

from pydantic import BaseModel


class RecommendationExplanationSchema(BaseModel):
    """Schema matching RecommendationExplanation outputs.

    Frontend behavior expectations:
    - Display champion name prominently.
    - Show explanation text as the primary narrative.
    - Optionally provide an expandable breakdown of numeric scores.
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
