"""Feature vector definitions for ML inference."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class FeatureVector:
    """Immutable feature vector used by recommendation models."""

    values: list[float] = field(default_factory=list)
