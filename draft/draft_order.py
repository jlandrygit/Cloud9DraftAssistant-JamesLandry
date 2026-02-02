"""Draft order utilities aligned with Riot competitive draft rules.

NOTE: This module is duplicated in core/draft_order.py for the ML training path.
Both modules contain identical DRAFT_SEQUENCE and helper functions. They are kept
separate to maintain clear boundaries between active inference code (draft/) and
ML training code (core/). If consolidating, ensure both code paths are updated.
"""

from __future__ import annotations

from typing import Iterable


# Ordered sequence of competitive draft steps (ban/pick order).
DRAFT_SEQUENCE: list[tuple[int, str, str, str]] = [
    (1, "BLUE", "BAN", "BB1"),
    (2, "RED", "BAN", "RB1"),
    (3, "BLUE", "BAN", "BB2"),
    (4, "RED", "BAN", "RB2"),
    (5, "BLUE", "BAN", "BB3"),
    (6, "RED", "BAN", "RB3"),
    (7, "BLUE", "PICK", "BP1"),
    (8, "RED", "PICK", "RP1"),
    (9, "RED", "PICK", "RP2"),
    (10, "BLUE", "PICK", "BP2"),
    (11, "BLUE", "PICK", "BP3"),
    (12, "RED", "PICK", "RP3"),
    (13, "RED", "BAN", "RB4"),
    (14, "BLUE", "BAN", "BB4"),
    (15, "RED", "BAN", "RB5"),
    (16, "BLUE", "BAN", "BB5"),
    (17, "RED", "PICK", "RP4"),
    (18, "BLUE", "PICK", "BP4"),
    (19, "BLUE", "PICK", "BP5"),
    (20, "RED", "PICK", "RP5"),
]


def is_pick(step: tuple[int, str, str, str]) -> bool:
    """Return True when the draft step is a pick."""
    return step[2] == "PICK"


def is_ban(step: tuple[int, str, str, str]) -> bool:
    """Return True when the draft step is a ban."""
    return step[2] == "BAN"


def acting_side(step: tuple[int, str, str, str]) -> str:
    """Return the side that acts on a given draft step."""
    return step[1]
