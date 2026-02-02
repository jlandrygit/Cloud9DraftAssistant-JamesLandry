"""Convert draft rows into step-level decision data."""

from __future__ import annotations

from typing import Any

import pandas as pd

from draft.draft_order import DRAFT_SEQUENCE, acting_side, is_ban, is_pick
from draft.draft_state import DraftState


def explode_draft_row(row: pd.Series, all_champions: set[str]) -> list[dict[str, Any]]:
    """Explode a single draft row into step-level decisions."""
    # Extract winner if available (optional)
    winner = str(row.get("WINNER", "")) if "WINNER" in row else ""
    
    state = DraftState(
        patch=str(row["PATCH"]),
        league=str(row["LEAGUE"]),
        blue_team=str(row["BLUE SIDE"]),
        red_team=str(row["RED SIDE"]),
        blue_picks=set(),
        red_picks=set(),
        bans=set(),
        step_index=0,
    )

    records: list[dict[str, Any]] = []
    for step in DRAFT_SEQUENCE:
        _, side, action_type, column_name = step
        action_champion = str(row[column_name]) if column_name in row else ""

        draft_state_dict = {
            "patch": state.patch,
            "league": state.league,
            "blue_team": state.blue_team,
            "red_team": state.red_team,
            "blue_picks": sorted(state.blue_picks),
            "red_picks": sorted(state.red_picks),
            "bans": sorted(state.bans),
            "step_index": state.step_index,
            "available_champions": sorted(
                state.available_champions(all_champions)
            ),
        }
        
        # Add winner information if available
        if winner:
            draft_state_dict["winner"] = winner

        records.append(
            {
                "draft_state": draft_state_dict,
                "action_champion": action_champion,
                "action_type": action_type,
                "acting_side": side,
                "step_index": state.step_index,
            }
        )

        if action_champion:
            state = state.advance(action_champion)
        else:
            # Advance even if missing so indexing remains consistent.
            state = DraftState(
                patch=state.patch,
                league=state.league,
                blue_team=state.blue_team,
                red_team=state.red_team,
                blue_picks=state.blue_picks,
                red_picks=state.red_picks,
                bans=state.bans,
                step_index=state.step_index + 1,
            )

    return records


def explode_all_drafts(
    df: pd.DataFrame, all_champions: set[str]
) -> pd.DataFrame:
    """Explode all draft rows into step-level decision data."""
    all_records: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        all_records.extend(explode_draft_row(row, all_champions))
    return pd.DataFrame(all_records)
