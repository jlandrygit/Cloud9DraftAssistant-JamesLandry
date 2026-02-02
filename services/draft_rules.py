"""Simplified draft rules for a hackathon demo.

This module encodes a deterministic, easy-to-reason-about draft order and
enforces basic legality checks (turn ownership, pick vs ban, no duplicates).
It intentionally avoids full professional draft complexity.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable

from core.draft_state import BanPhase, BanPhaseBans, DraftPhase, DraftState, Role, Side


class ActionType(str, Enum):
    """Supported draft action types."""

    PICK = "pick"
    BAN = "ban"


@dataclass(frozen=True)
class LegalAction:
    """A legal draft action produced by the rules engine."""

    action_type: ActionType
    side: Side
    champion_id: str
    role: Role | None = None


@dataclass(frozen=True)
class DraftStep:
    """Single step in the simplified draft order."""

    phase: DraftPhase
    action_type: ActionType
    side: Side


class DraftRules:
    """Minimal draft rules engine for demo use.

    This class defines a fixed draft order and validates the current turn
    against that order. Picks use the next open role in a fixed role sequence.
    """

    _ROLE_ORDER = (Role.TOP, Role.JUNGLE, Role.MID, Role.BOTTOM, Role.SUPPORT)
    _DRAFT_ORDER = (
        # Ban phase 1: blue/red alternating, 3 bans each.
        DraftStep(DraftPhase.BAN_PHASE_1, ActionType.BAN, Side.BLUE),
        DraftStep(DraftPhase.BAN_PHASE_1, ActionType.BAN, Side.RED),
        DraftStep(DraftPhase.BAN_PHASE_1, ActionType.BAN, Side.BLUE),
        DraftStep(DraftPhase.BAN_PHASE_1, ActionType.BAN, Side.RED),
        DraftStep(DraftPhase.BAN_PHASE_1, ActionType.BAN, Side.BLUE),
        DraftStep(DraftPhase.BAN_PHASE_1, ActionType.BAN, Side.RED),
        # Pick phase 1: blue, red, red, blue.
        DraftStep(DraftPhase.PICK_PHASE_1, ActionType.PICK, Side.BLUE),
        DraftStep(DraftPhase.PICK_PHASE_1, ActionType.PICK, Side.RED),
        DraftStep(DraftPhase.PICK_PHASE_1, ActionType.PICK, Side.RED),
        DraftStep(DraftPhase.PICK_PHASE_1, ActionType.PICK, Side.BLUE),
        # Ban phase 2: blue/red alternating, 2 bans each.
        DraftStep(DraftPhase.BAN_PHASE_2, ActionType.BAN, Side.BLUE),
        DraftStep(DraftPhase.BAN_PHASE_2, ActionType.BAN, Side.RED),
        DraftStep(DraftPhase.BAN_PHASE_2, ActionType.BAN, Side.BLUE),
        DraftStep(DraftPhase.BAN_PHASE_2, ActionType.BAN, Side.RED),
        # Pick phase 2: red, blue, blue, red.
        DraftStep(DraftPhase.PICK_PHASE_2, ActionType.PICK, Side.RED),
        DraftStep(DraftPhase.PICK_PHASE_2, ActionType.PICK, Side.BLUE),
        DraftStep(DraftPhase.PICK_PHASE_2, ActionType.PICK, Side.BLUE),
        DraftStep(DraftPhase.PICK_PHASE_2, ActionType.PICK, Side.RED),
    )

    def __init__(self, champion_pool: Iterable[str]) -> None:
        self._champion_pool = list(champion_pool)

    def enumerate_legal_actions(self, state: DraftState) -> list[LegalAction]:
        """Return all legal actions for the current state."""
        step = self._current_step(state)
        used = self._used_champions(state)
        available = [champ for champ in self._champion_pool if champ not in used]

        if step.action_type is ActionType.BAN:
            return [
                LegalAction(
                    action_type=ActionType.BAN,
                    side=step.side,
                    champion_id=champion_id,
                )
                for champion_id in available
            ]

        role = self._next_open_role(state, step.side)
        if role is None:
            return []
        return [
            LegalAction(
                action_type=ActionType.PICK,
                side=step.side,
                champion_id=champion_id,
                role=role,
            )
            for champion_id in available
        ]

    def apply_action(self, state: DraftState, action: LegalAction) -> DraftState:
        """Apply a legal action and return the next DraftState."""
        step = self._current_step(state)
        if state.phase is not step.phase:
            raise ValueError("Draft phase does not match expected phase for turn.")
        if action.action_type is not step.action_type:
            raise ValueError("Action type is not legal for the current phase.")
        if action.side is not step.side:
            raise ValueError("Action side does not match the current turn owner.")
        if action.champion_id in self._used_champions(state):
            raise ValueError("Champion has already been picked or banned.")

        if action.action_type is ActionType.PICK:
            expected_role = self._next_open_role(state, action.side)
            if action.role is None or action.role is not expected_role:
                raise ValueError("Pick role is not legal for the current turn.")
            next_state = state.with_pick(action.side, action.role, action.champion_id)
        else:
            ban_phase = self._ban_phase_for(step.phase)
            if ban_phase is None:
                raise ValueError("Ban action is not legal for the current phase.")
            next_state = state.with_ban(
                ban_phase, action.side, action.champion_id
            )

        next_turn = state.turn + 1
        next_phase = (
            self._DRAFT_ORDER[next_turn].phase
            if next_turn < len(self._DRAFT_ORDER)
            else state.phase
        )
        return next_state.with_phase(next_phase, next_turn)

    def _current_step(self, state: DraftState) -> DraftStep:
        """Return the draft step for the current turn."""
        if state.turn < 0 or state.turn >= len(self._DRAFT_ORDER):
            raise ValueError("Draft turn is out of range for the configured order.")
        step = self._DRAFT_ORDER[state.turn]
        if step.phase is not state.phase:
            raise ValueError("Draft phase does not align with configured order.")
        return step

    def _next_open_role(self, state: DraftState, side: Side) -> Role | None:
        """Return the next unfilled role for the given side."""
        picks = state.blue_picks if side is Side.BLUE else state.red_picks
        pick_map = picks.to_dict()
        for role in self._ROLE_ORDER:
            if pick_map.get(role.value) is None:
                return role
        return None

    def _used_champions(self, state: DraftState) -> set[str]:
        """Collect all picked and banned champions."""
        picks = list(state.blue_picks.to_dict().values()) + list(
            state.red_picks.to_dict().values()
        )
        bans = []
        for ban_phase in self._ensure_ban_phases(state).ban_phases:
            bans.extend(ban_phase.blue_bans)
            bans.extend(ban_phase.red_bans)
        return {champion for champion in picks + bans if champion}

    def _ban_phase_for(self, phase: DraftPhase) -> BanPhase | None:
        """Map draft phase to ban phase if applicable."""
        if phase is DraftPhase.BAN_PHASE_1:
            return BanPhase.BAN_PHASE_1
        if phase is DraftPhase.BAN_PHASE_2:
            return BanPhase.BAN_PHASE_2
        if phase is DraftPhase.BAN_PHASE_3:
            return BanPhase.BAN_PHASE_3
        return None

    def _ensure_ban_phases(self, state: DraftState) -> DraftState:
        """Ensure the state has ban phases to avoid empty tracking."""
        if state.ban_phases:
            return state
        ban_phases = (
            BanPhaseBans(phase=BanPhase.BAN_PHASE_1),
            BanPhaseBans(phase=BanPhase.BAN_PHASE_2),
            BanPhaseBans(phase=BanPhase.BAN_PHASE_3),
        )
        return DraftState(
            blue_team=state.blue_team,
            red_team=state.red_team,
            phase=state.phase,
            turn=state.turn,
            blue_picks=state.blue_picks,
            red_picks=state.red_picks,
            ban_phases=ban_phases,
        )
