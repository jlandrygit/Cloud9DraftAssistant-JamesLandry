"""Utilities for loading draft data from Excel."""

from __future__ import annotations

from typing import Iterable

import pandas as pd


REQUIRED_COLUMNS = [
    "PATCH",
    "LEAGUE",
    "BLUE SIDE",
    "RED SIDE",
    "BB1",
    "RB1",
    "BB2",
    "RB2",
    "BB3",
    "RB3",
    "BP1",
    "RP1",
    "RP2",
    "BP2",
    "BP3",
    "RP3",
    "RB4",
    "BB4",
    "RB5",
    "BB5",
    "RP4",
    "BP4",
    "BP5",
    "RP5",
]

# New format columns (with First Pick/Last Pick instead of Blue Side/Red Side)
NEW_FORMAT_COLUMNS = [
    "PATCH",
    "LEAGUE",
    "FIRST PICK",
    "LAST PICK",
    "BB1",
    "RB1",
    "BB2",
    "RB2",
    "BB3",
    "RB3",
    "BP1",
    "RP1",
    "RP2",
    "BP2",
    "BP3",
    "RP3",
    "RB4",
    "BB4",
    "RB5",
    "BB5",
    "RP4",
    "BP4",
    "BP5",
    "RP5",
]

# Optional columns (won't cause error if missing)
OPTIONAL_COLUMNS = ["WINNER"]


def load_draft_excel(path: str) -> pd.DataFrame:
    """Load a draft Excel file and validate draft ordering columns.

    Supports two formats:
    1. Old format: "BLUE SIDE", "RED SIDE" columns
    2. New format: "FIRST PICK", "LAST PICK" columns (maps to BLUE/RED)

    Draft ordering assumptions:
    - BB1 means First Pick Team Ban 1 (mapped to BLUE), RB1 means Last Pick Team Ban 1 (mapped to RED), etc.
    - BP1 means First Pick Team Pick 1 (mapped to BLUE), RP1 means Last Pick Team Pick 1 (mapped to RED), etc.
    - The required columns reflect a standard competitive ban/pick sequence.
    """
    df = pd.read_excel(path)
    df = _normalize_columns(df)
    
    # Check which format we have
    has_old_format = "BLUE SIDE" in df.columns and "RED SIDE" in df.columns
    has_new_format = "FIRST PICK" in df.columns and "LAST PICK" in df.columns
    
    if has_old_format:
        # Old format - ensure required columns
        _ensure_columns(df, REQUIRED_COLUMNS)
    elif has_new_format:
        # New format - ensure required columns and convert to old format
        _ensure_columns(df, NEW_FORMAT_COLUMNS)
        # Map First Pick -> Blue Side, Last Pick -> Red Side
        df["BLUE SIDE"] = df["FIRST PICK"]
        df["RED SIDE"] = df["LAST PICK"]
    else:
        raise ValueError(
            "Dataset must have either ('BLUE SIDE', 'RED SIDE') or ('FIRST PICK', 'LAST PICK') columns"
        )
    
    return df


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names for consistent casing and spacing."""
    result = df.copy()
    result.columns = [str(col).strip().upper() for col in result.columns]
    return result


def _ensure_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    """Raise ValueError when required columns are missing."""
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            "Missing required draft columns: " + ", ".join(missing)
        )
