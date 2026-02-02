"""Centralized configuration for draft scoring and inference.

This module consolidates all magic numbers, thresholds, weights, and default values
used throughout the draft recommendation system. All configuration values include
comments explaining why they were chosen.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


# ============================================================================
# Patch and League Configuration
# ============================================================================

DEFAULT_PATCH_VERSION = "26.2"
"""Default patch version for draft state.
Chosen to match the current competitive patch at hackathon time.
"""

DEFAULT_LEAGUE = "LCS"
"""Default league identifier for demo mode.
Used when no specific league context is provided.
"""

MODEL_PATCH_START = os.getenv("MODEL_PATCH_START", "25.16")
"""Start patch for model training data.
Lower bound of patch range used to train the policy model.
"""

MODEL_PATCH_END = os.getenv("MODEL_PATCH_END", "26.2")
"""End patch for model training data.
Upper bound of patch range used to train the policy model.
"""


# ============================================================================
# Model Scoring Weights
# ============================================================================

POLICY_WEIGHT = 0.60
"""Weight for expert draft policy model score.
Primary decision driver based on professional draft patterns.
Set to 60% to maintain policy dominance while allowing modifiers.
"""

COMFORT_WEIGHT = 0.05
"""Weight for player comfort scores.
Low weight because comfort is a secondary modifier, not a primary decision factor.
Balances player preference with meta strength.
"""

COUNTER_WEIGHT = 0.05
"""Weight for counter-pick scores.
Low weight because counter-picking is situational and context-dependent.
Prevents over-reliance on counter data which may be noisy.
"""

SYNERGY_WEIGHT = 0.13
"""Weight for team synergy scores.
Higher than comfort/counter because synergy affects entire team composition.
Represents the importance of champion combinations in draft success.
"""

META_WEIGHT = 0.12
"""Weight for patch meta strength scores.
Balances patch-specific power with policy patterns.
Allows meta shifts to influence recommendations without overriding expert patterns.
"""


# ============================================================================
# Scoring Thresholds
# ============================================================================

POLICY_HIGH_THRESHOLD = 0.9
"""Policy score threshold for high-confidence recommendations.
When policy score >= 0.9, use policy message as primary explanation.
Chosen to identify champions with very strong professional draft support.
"""

FACTOR_SCORE_THRESHOLD = 0.5
"""Minimum score threshold for non-policy factors to be included in explanations.
Filters out weak signals to keep explanations focused and relevant.
Prevents noise from low-confidence factors from cluttering explanations.
"""

CONFIDENCE_HIGH_THRESHOLD = 0.7
"""Score threshold for "High" confidence label.
Absolute threshold based on model output distribution analysis.
Chosen to represent top ~30% of recommendations.
"""

CONFIDENCE_MEDIUM_THRESHOLD = 0.4
"""Score threshold for "Medium" confidence label.
Absolute threshold for mid-tier recommendations.
Separates clearly strong picks from marginal ones.
"""

BAN_RATE_COMFORT_THRESHOLD = 0.4
"""Ban rate threshold for triggering comfort messages.
When a champion is banned >= 40% of the time against a team, it's a strong signal.
Indicates the champion is a significant threat for that team.
"""


# ============================================================================
# Player Comfort Configuration
# ============================================================================

COMFORT_GAMES_THRESHOLD = 20
"""Number of games required for maximum comfort score.
Based on analysis showing 20+ games provides stable performance estimates.
Below this threshold, comfort scores scale linearly from 0 to 1.
"""


# ============================================================================
# Match Data Confidence Thresholds
# ============================================================================

MATCH_CONFIDENCE_THRESHOLD = 250
"""Minimum matches required for full confidence weighting in synergy/counter scores.
Below this threshold, scores are down-weighted proportionally.
Chosen to ensure statistical significance while allowing newer champions to be scored.
"""

WINRATE_BASELINE = 45.0
"""Baseline winrate for normalizing counter/synergy scores.
"""

WINRATE_SCALING_FACTOR = 10.0
"""Scaling factor for winrate normalization.
Converts winrate difference from baseline into a 0-1 score range.
Formula: (winrate - baseline) / scaling_factor
"""


# ============================================================================
# Ban Rate Bonus Configuration
# ============================================================================

BAN_RATE_TOTAL_GAMES = 246
"""Total games used for calculating ban rate percentages.
Matches the dataset size for pro meta statistics.
Used to normalize ban counts into percentages.
"""

BAN_RATE_TIER_1_THRESHOLD = 80.0
"""Ban rate threshold for tier 1 bonus (1.5x-2.0x multiplier).
Champions with 80%+ ban rate are extremely meta-defining.
Rewards picking these when available despite low pick counts.
"""

BAN_RATE_TIER_2_THRESHOLD = 50.0
"""Ban rate threshold for tier 2 bonus (1.2x-1.5x multiplier).
Champions with 50-80% ban rate are strong meta picks.
Moderate bonus to reflect their power level.
"""

BAN_RATE_TIER_3_THRESHOLD = 25.0
"""Ban rate threshold for tier 3 bonus (1.05x-1.2x multiplier).
Champions with 25-50% ban rate are viable meta options.
Small bonus to acknowledge their strength.
"""

BAN_RATE_TIER_4_THRESHOLD = 10.0
"""Ban rate threshold for tier 4 bonus (1.0x-1.05x multiplier).
Champions with 10-25% ban rate are occasionally banned.
Minimal bonus for edge cases.
"""


# ============================================================================
# Pro Play Frequency Penalties
# ============================================================================

FREQUENCY_PENALTY_ZERO_PRESENCE = 0.1
"""Multiplier for champions with zero presence (picks + bans).
90% penalty prevents recommending champions never seen in pro play.
Assumes zero presence indicates the champion is not viable at pro level.
"""

FREQUENCY_PENALTY_ONE_PRESENCE = 0.25
"""Multiplier for champions with exactly one presence.
75% penalty for champions with minimal pro play exposure.
Starting point for scaling up to full multiplier.
"""

FREQUENCY_PENALTY_MAX_PRESENCE = 50
"""Presence count required for full multiplier (1.0).
Champions with 50+ presence are well-established in pro meta.
Scales linearly from 1 presence (0.25) to 50+ presence (1.0).
"""


# ============================================================================
# Role Denial Bonus Configuration
# ============================================================================

ROLE_DENIAL_BONUS_BASE = 0.05
"""Base bonus for role denial when no champions in role are gone yet.
Small incentive to start "pinching" a role (banning multiple champions in same role).
Encourages strategic role targeting in second ban phase.
"""

ROLE_DENIAL_BONUS_ONE_GONE = 0.10
"""Bonus when 1 champion in the target role has been picked/banned.
Moderate incentive to continue pinching the role.
"""

ROLE_DENIAL_BONUS_TWO_GONE = 0.15
"""Bonus when 2 champions in the target role have been picked/banned.
Strong incentive as the role pool starts to thin.
"""

ROLE_DENIAL_BONUS_MAX = 0.20
"""Maximum bonus for role denial (3+ champions gone).
Caps at 0.20 to prevent excessive bonus stacking.
Represents maximum strategic value of role pinching.
"""

ROLE_DENIAL_BONUS_INCREMENT = 0.025
"""Incremental bonus per additional champion beyond 2.
Small linear increase for each additional champion removed from role.
Prevents exponential scaling while rewarding continued pinching.
"""


# ============================================================================
# Archetype Bonus Configuration
# ============================================================================

ARCHETYPE_BONUS_BASE = 0.05
"""Base bonus for matching team archetype or countering opponent archetype.
Starting point for archetype-based synergy/counter bonuses.
"""

ARCHETYPE_BONUS_INCREMENT = 0.05
"""Incremental bonus per additional matching champion.
Scales from 0.05 (1 match) to 0.15 (3+ matches).
Rewards stronger archetype alignment.
"""

ARCHETYPE_BONUS_MAX = 0.15
"""Maximum archetype bonus (3+ matching champions).
Caps at 0.15 to prevent over-weighting archetype vs other factors.
Represents maximum value of archetype synergy/counter.
"""


# ============================================================================
# Model Inference Configuration
# ============================================================================

LOGIT_CLIP_MIN = -20.0
"""Minimum value for logit clipping before sigmoid.
Prevents numerical underflow in sigmoid calculation.
Standard practice for numerical stability.
"""

LOGIT_CLIP_MAX = 20.0
"""Maximum value for logit clipping before sigmoid.
Prevents numerical overflow in sigmoid calculation.
Standard practice for numerical stability.
"""


# ============================================================================
# UI Configuration
# ============================================================================

CHAMPION_SEARCH_LIMIT = 20
"""Maximum number of champion search results to display.
Prevents UI clutter while providing sufficient options.
Balances usability with performance.
"""


# ============================================================================
# Legacy Settings (for FastAPI service)
# ============================================================================

@dataclass(frozen=True)
class Settings:
    """Runtime settings for the FastAPI service."""

    app_name: str
    environment: str


def load_settings() -> Settings:
    """Load settings from environment variables with safe defaults."""
    return Settings(
        app_name=os.getenv("APP_NAME", "C9 Draft Assistant"),
        environment=os.getenv("ENVIRONMENT", "development"),
    )
