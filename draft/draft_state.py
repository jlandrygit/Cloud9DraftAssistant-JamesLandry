"""Immutable draft state for competitive pick/ban flow."""

from __future__ import annotations

from dataclasses import dataclass, replace

from draft.draft_order import DRAFT_SEQUENCE, acting_side, is_ban, is_pick


@dataclass(frozen=True)
class DraftState:
    """Immutable draft state snapshot.

    This class does not validate legality. Draft rules/validation should be
    enforced by a separate rules engine before advancing the state.
    """

    patch: str
    league: str
    blue_team: str
    red_team: str
    blue_picks: set[str]
    red_picks: set[str]
    bans: set[str]
    step_index: int

    def available_champions(self, all_champions: set[str]) -> set[str]:
        """Return the champions not yet picked or banned."""
        used = self.blue_picks | self.red_picks | self.bans
        return set(all_champions) - used

    def is_pick_phase(self) -> bool:
        """Return True if the current step is a pick."""
        return is_pick(DRAFT_SEQUENCE[self.step_index])

    def is_ban_phase(self) -> bool:
        """Return True if the current step is a ban."""
        return is_ban(DRAFT_SEQUENCE[self.step_index])

    def acting_side(self) -> str:
        """Return the side that must act on the current step."""
        return acting_side(DRAFT_SEQUENCE[self.step_index])

    def advance(self, action_champion: str) -> DraftState:
        """Advance the draft state with a pick or ban.

        Legality is not checked here; callers must validate the action first.
        """
        step = DRAFT_SEQUENCE[self.step_index]
        side = acting_side(step)
        if is_pick(step):
            if side == "BLUE":
                next_state = replace(
                    self, blue_picks=self.blue_picks | {action_champion}
                )
            else:
                next_state = replace(
                    self, red_picks=self.red_picks | {action_champion}
                )
        else:
            next_state = replace(self, bans=self.bans | {action_champion})

        return replace(next_state, step_index=self.step_index + 1)
