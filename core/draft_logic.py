"""Draft state machine for a standard pro draft order."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Side(str, Enum):
    """Draft side assignment."""

    BLUE = "Blue"
    RED = "Red"


class ActionType(str, Enum):
    """Draft action type."""

    PICK = "Pick"
    BAN = "Ban"


@dataclass(frozen=True)
class DraftStep:
    """Single step in the competitive draft order."""

    phase_label: str
    action_type: ActionType
    side: Side


DRAFT_ORDER: tuple[DraftStep, ...] = (
    # Ban phase 1 (3 bans each).
    DraftStep("Blue Ban 1", ActionType.BAN, Side.BLUE),
    DraftStep("Red Ban 1", ActionType.BAN, Side.RED),
    DraftStep("Blue Ban 2", ActionType.BAN, Side.BLUE),
    DraftStep("Red Ban 2", ActionType.BAN, Side.RED),
    DraftStep("Blue Ban 3", ActionType.BAN, Side.BLUE),
    DraftStep("Red Ban 3", ActionType.BAN, Side.RED),
    # Pick phase 1 (blue, red, red, blue, blue, red).
    DraftStep("Blue Pick 1", ActionType.PICK, Side.BLUE),
    DraftStep("Red Pick 1", ActionType.PICK, Side.RED),
    DraftStep("Red Pick 2", ActionType.PICK, Side.RED),
    DraftStep("Blue Pick 2", ActionType.PICK, Side.BLUE),
    DraftStep("Blue Pick 3", ActionType.PICK, Side.BLUE),
    DraftStep("Red Pick 3", ActionType.PICK, Side.RED),
    # Ban phase 2 (red, blue, red, blue).
    DraftStep("Red Ban 4", ActionType.BAN, Side.RED),
    DraftStep("Blue Ban 4", ActionType.BAN, Side.BLUE),
    DraftStep("Red Ban 5", ActionType.BAN, Side.RED),
    DraftStep("Blue Ban 5", ActionType.BAN, Side.BLUE),
    # Pick phase 2 (red, blue, blue, red).
    DraftStep("Red Pick 4", ActionType.PICK, Side.RED),
    DraftStep("Blue Pick 4", ActionType.PICK, Side.BLUE),
    DraftStep("Blue Pick 5", ActionType.PICK, Side.BLUE),
    DraftStep("Red Pick 5", ActionType.PICK, Side.RED),
)


@dataclass
class DraftState:
    """Mutable draft state for a live, turn-based draft."""

    champion_pool: list[str]
    step_index: int = 0
    blue_bans: list[str] = field(default_factory=list)
    red_bans: list[str] = field(default_factory=list)
    blue_picks: list[str] = field(default_factory=list)
    red_picks: list[str] = field(default_factory=list)

    @property
    def current_phase(self) -> str:
        """Current phase label from the draft order."""
        if self.is_complete:
            return "Complete"
        return DRAFT_ORDER[self.step_index].phase_label

    @property
    def current_turn(self) -> Side:
        """Side that must act on the current turn."""
        if self.is_complete:
            return DRAFT_ORDER[-1].side
        return DRAFT_ORDER[self.step_index].side

    @property
    def current_action(self) -> ActionType:
        """Action type required for the current turn."""
        if self.is_complete:
            return DRAFT_ORDER[-1].action_type
        return DRAFT_ORDER[self.step_index].action_type

    @property
    def available_champions(self) -> list[str]:
        """Champions not yet picked or banned."""
        used = set(self.blue_bans + self.red_bans + self.blue_picks + self.red_picks)
        return [champion for champion in self.champion_pool if champion not in used]

    @property
    def is_complete(self) -> bool:
        """Whether the draft has finished."""
        return self.step_index >= len(DRAFT_ORDER)

    def can_take_action(
        self, side: Side, action_type: ActionType, champion: str
    ) -> tuple[bool, str]:
        """Validate whether an action is legal for the current state."""
        if self.is_complete:
            return False, "Draft is complete."
        step = DRAFT_ORDER[self.step_index]
        if step.side is not side:
            return False, f"It is {step.side.value}'s turn."
        if step.action_type is not action_type:
            return False, f"Current phase requires a {step.action_type.value.lower()}."
        if champion not in self.champion_pool:
            return False, "Champion is not in the draft pool."
        if champion in self.blue_bans or champion in self.red_bans:
            return False, "Champion is already banned."
        if champion in self.blue_picks or champion in self.red_picks:
            return False, "Champion is already picked."
        return True, ""

    def apply_action(self, side: Side, action_type: ActionType, champion: str) -> None:
        """Apply a legal action to the current draft state."""
        is_valid, reason = self.can_take_action(side, action_type, champion)
        if not is_valid:
            raise ValueError(reason)
        if action_type is ActionType.BAN:
            if side is Side.BLUE:
                self.blue_bans.append(champion)
            else:
                self.red_bans.append(champion)
        else:
            if side is Side.BLUE:
                self.blue_picks.append(champion)
            else:
                self.red_picks.append(champion)
        advance_phase(self)


def advance_phase(state: DraftState) -> None:
    """Advance the draft to the next step in the order."""
    if not state.is_complete:
        state.step_index += 1
