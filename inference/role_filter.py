"""Role-based filtering for draft recommendations."""

from __future__ import annotations

from functools import lru_cache
from os import getenv

from data.meta.load_u_gg_roles import load_roles
from draft.draft_state import DraftState

ALL_ROLES = {"TOP", "JUNGLE", "MID", "ADC", "SUPPORT"}


def filter_by_filled_roles(state: DraftState, candidates: set[str]) -> set[str]:
    """Filter out champions whose possible roles are already filled.

    This uses a combinatorial check over current picks: a candidate is removed
    only if every possible assignment of existing picks to roles already covers
    all of the candidate's roles.
    """
    roles_map = _load_roles()
    if not roles_map:
        return candidates

    role_options = _get_role_options(state, roles_map)
    if not role_options:
        return candidates
    possible_filled = _possible_filled_role_sets(role_options)
    if not possible_filled:
        # If no valid assignment exists, avoid over-filtering.
        return candidates

    filtered: set[str] = set()
    for champion in candidates:
        champ_roles = roles_map.get(champion)
        if not champ_roles:
            filtered.add(champion)
            continue
        # Remove only if all assignments already cover every candidate role.
        if _roles_always_filled(set(champ_roles), possible_filled):
            continue
        filtered.add(champion)
    return filtered


def _get_role_options(
    state: DraftState, roles_map: dict[str, set[str]]
) -> list[set[str]]:
    """Return role options for the relevant team's existing picks."""
    if state.is_pick_phase():
        picks = state.blue_picks if state.acting_side() == "BLUE" else state.red_picks
    else:
        # For bans, evaluate the opposing team's already-picked roles.
        picks = state.red_picks if state.acting_side() == "BLUE" else state.blue_picks

    options: list[set[str]] = []
    for champion in picks:
        roles = roles_map.get(champion)
        if not roles:
            # Unknown roles -> be permissive to avoid over-filtering.
            options.append(set(ALL_ROLES))
        else:
            options.append(set(roles))
    return options


def get_open_roles(
    state: DraftState, roles_map: dict[str, set[str]], side: str
) -> set[str]:
    """Return roles that are still open for the specified side."""
    if side == "BLUE":
        picks = state.blue_picks
    else:
        picks = state.red_picks
    role_options: list[set[str]] = []
    for champion in picks:
        roles = roles_map.get(champion)
        if not roles:
            role_options.append(set(ALL_ROLES))
        else:
            role_options.append(set(roles))
    if not role_options:
        return set(ALL_ROLES)
    possible = _possible_filled_role_sets(role_options)
    if not possible:
        return set(ALL_ROLES)
    always_filled = set.intersection(*(set(item) for item in possible))
    return set(ALL_ROLES) - set(always_filled)


def _possible_filled_role_sets(role_options: list[set[str]]) -> set[frozenset[str]]:
    """Return all possible role sets that current picks could fill."""
    if not role_options:
        return set()

    possible: set[frozenset[str]] = set()

    def backtrack(index: int, used_roles: set[str]) -> None:
        if index >= len(role_options):
            possible.add(frozenset(used_roles))
            return
        options = role_options[index]
        for role in options:
            if role in used_roles:
                continue
            used_roles.add(role)
            backtrack(index + 1, used_roles)
            used_roles.remove(role)

    backtrack(0, set())
    return possible


def _roles_always_filled(
    candidate_roles: set[str], possible_filled: set[frozenset[str]]
) -> bool:
    """Return True if candidate roles are covered in every assignment."""
    for filled in possible_filled:
        if not candidate_roles.issubset(filled):
            return False
    return True


@lru_cache(maxsize=1)
def _load_roles() -> dict[str, set[str]]:
    """Load champion roles from disk once per process."""
    path = getenv("UGG_ROLES_PATH", "data/meta/u_gg_roles.csv")
    try:
        return load_roles(path)
    except Exception:
        return {}
