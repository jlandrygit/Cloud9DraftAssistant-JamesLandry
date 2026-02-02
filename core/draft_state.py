"""Draft state data structures for a live League of Legends draft."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any


class Side(str, Enum):
    """Draft side assignment."""

    BLUE = "blue"
    RED = "red"


class Role(str, Enum):
    """Standard League of Legends roles."""

    TOP = "top"
    JUNGLE = "jungle"
    MID = "mid"
    BOTTOM = "bottom"
    SUPPORT = "support"


class DraftPhase(str, Enum):
    """High-level phases of the live draft."""

    BAN_PHASE_1 = "ban_phase_1"
    PICK_PHASE_1 = "pick_phase_1"
    BAN_PHASE_2 = "ban_phase_2"
    PICK_PHASE_2 = "pick_phase_2"
    BAN_PHASE_3 = "ban_phase_3"
    PICK_PHASE_3 = "pick_phase_3"

    def is_ban(self) -> bool:
        """Return True when the phase is a ban phase."""
        return self.value.startswith("ban_")


class BanPhase(str, Enum):
    """Only ban phases, used for tracking bans per phase."""

    BAN_PHASE_1 = "ban_phase_1"
    BAN_PHASE_2 = "ban_phase_2"
    BAN_PHASE_3 = "ban_phase_3"


@dataclass(frozen=True)
class RolePicks:
    """Role-based picks for a single side."""

    top: str | None = None
    jungle: str | None = None
    mid: str | None = None
    bottom: str | None = None
    support: str | None = None

    def to_dict(self) -> dict[str, str | None]:
        """Serialize role picks to a JSON-friendly dict."""
        return {
            Role.TOP.value: self.top,
            Role.JUNGLE.value: self.jungle,
            Role.MID.value: self.mid,
            Role.BOTTOM.value: self.bottom,
            Role.SUPPORT.value: self.support,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, str | None]) -> RolePicks:
        """Deserialize role picks from JSON-friendly data."""
        return cls(
            top=payload.get(Role.TOP.value),
            jungle=payload.get(Role.JUNGLE.value),
            mid=payload.get(Role.MID.value),
            bottom=payload.get(Role.BOTTOM.value),
            support=payload.get(Role.SUPPORT.value),
        )


@dataclass(frozen=True)
class BanPhaseBans:
    """Bans for a single ban phase for both sides."""

    phase: BanPhase
    blue_bans: tuple[str, ...] = ()
    red_bans: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Serialize bans for this phase."""
        return {
            "phase": self.phase.value,
            Side.BLUE.value: list(self.blue_bans),
            Side.RED.value: list(self.red_bans),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> BanPhaseBans:
        """Deserialize bans for this phase."""
        return cls(
            phase=BanPhase(payload["phase"]),
            blue_bans=tuple(payload.get(Side.BLUE.value, [])),
            red_bans=tuple(payload.get(Side.RED.value, [])),
        )


@dataclass(frozen=True)
class DraftState:
    """Immutable snapshot of a live draft state.

    This model is intended to be updated by draft events (ban/pick actions)
    and then passed through services for recommendations or validation.
    Each update returns a new DraftState instance, preserving the prior
    snapshot for audit or replay.
    """

    blue_team: str
    red_team: str
    phase: DraftPhase
    turn: int
    blue_picks: RolePicks
    red_picks: RolePicks
    ban_phases: tuple[BanPhaseBans, ...]

    def to_dict(self) -> dict[str, Any]:
        """Serialize the draft state into JSON-friendly data."""
        return {
            "blue_team": self.blue_team,
            "red_team": self.red_team,
            "phase": self.phase.value,
            "turn": self.turn,
            "picks": {
                Side.BLUE.value: self.blue_picks.to_dict(),
                Side.RED.value: self.red_picks.to_dict(),
            },
            "bans": [ban_phase.to_dict() for ban_phase in self.ban_phases],
        }

    def to_json(self) -> str:
        """Serialize the draft state to a JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> DraftState:
        """Deserialize the draft state from JSON-friendly data."""
        return cls(
            blue_team=payload["blue_team"],
            red_team=payload["red_team"],
            phase=DraftPhase(payload["phase"]),
            turn=int(payload["turn"]),
            blue_picks=RolePicks.from_dict(payload["picks"][Side.BLUE.value]),
            red_picks=RolePicks.from_dict(payload["picks"][Side.RED.value]),
            ban_phases=tuple(
                BanPhaseBans.from_dict(ban_payload)
                for ban_payload in payload.get("bans", [])
            ),
        )

    @classmethod
    def from_json(cls, payload: str) -> DraftState:
        """Deserialize the draft state from a JSON string."""
        return cls.from_dict(json.loads(payload))

    def with_pick(self, side: Side, role: Role, champion_id: str) -> DraftState:
        """Return a new state with a pick recorded for the given side and role.

        Warning: this method does not validate draft legality. Use DraftRules
        to validate the action before calling this method.
        """
        if side is Side.BLUE:
            updated = replace(self.blue_picks, **{role.value: champion_id})
            return replace(self, blue_picks=updated)
        updated = replace(self.red_picks, **{role.value: champion_id})
        return replace(self, red_picks=updated)

    def with_ban(self, phase: BanPhase, side: Side, champion_id: str) -> DraftState:
        """Return a new state with a ban recorded for a specific phase.

        Warning: this method does not validate draft legality. Use DraftRules
        to validate the action before calling this method.
        """
        updated_phases: list[BanPhaseBans] = []
        for ban_phase in self.ban_phases:
            if ban_phase.phase is phase:
                if side is Side.BLUE:
                    updated_phases.append(
                        replace(ban_phase, blue_bans=ban_phase.blue_bans + (champion_id,))
                    )
                else:
                    updated_phases.append(
                        replace(ban_phase, red_bans=ban_phase.red_bans + (champion_id,))
                    )
            else:
                updated_phases.append(ban_phase)
        return replace(self, ban_phases=tuple(updated_phases))

    def with_phase(self, phase: DraftPhase, turn: int) -> DraftState:
        """Return a new state with updated phase and turn values."""
        return replace(self, phase=phase, turn=turn)
