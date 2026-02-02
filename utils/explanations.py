"""Helpers for turning recommendation metadata into coach-friendly text."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class RecommendationMetadata:
    """Structured metadata used to explain a recommendation."""

    champion_id: str
    score: float
    factors: Mapping[str, float]


def build_explanation(metadata: RecommendationMetadata) -> str:
    """Convert structured metadata into a short coach-facing explanation.

    The explanation is limited to two sentences and only references provided
    fields. It never fabricates details beyond the metadata.
    """
    factors = _top_factors(metadata.factors)
    if not factors:
        return f"{metadata.champion_id} is recommended based on the available data."

    parts = [f"{name}={value:.2f}" for name, value in factors]
    sentence_one = (
        f"{metadata.champion_id} ranks well due to " + ", ".join(parts) + "."
    )
    sentence_two = f"Overall score: {metadata.score:.2f}."
    return f"{sentence_one} {sentence_two}"


def _top_factors(factors: Mapping[str, float], limit: int = 2) -> list[tuple[str, float]]:
    """Pick the most influential factors by absolute value."""
    items = [(name, float(value)) for name, value in factors.items()]
    items.sort(key=lambda item: abs(item[1]), reverse=True)
    return items[:limit]
