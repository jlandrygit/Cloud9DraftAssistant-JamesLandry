"""Load and validate U.GG winrate data for the current patch."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_winrates(path: str) -> dict[str, dict[str, float]]:
    """Load champion meta stats and validate schema and patch.

    The CSV is expected to contain: patch, champion, role, winrate, pickrate, banrate.
    When multiple roles exist for a champion, the max values are used.
    """
    df = pd.read_csv(path)
    required = {"patch", "champion", "winrate", "pickrate", "banrate"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in winrate CSV: {sorted(missing)}")

    patch_values = df["patch"].dropna().unique()
    if len(patch_values) != 1:
        raise ValueError(f"Expected exactly one patch value, found: {patch_values}")
    patch = str(patch_values[0])
    _validate_patch(patch)

    stats: dict[str, dict[str, float]] = {}
    for champion, group in df.groupby("champion"):
        winrate = float(group["winrate"].astype(float).max())
        pickrate = float(group["pickrate"].astype(float).max())
        banrate = float(group["banrate"].astype(float).max())
        # All values in CSV are percentages, convert to decimals
        stats[str(champion)] = {
            "winrate": winrate / 100.0,
            "pickrate": pickrate / 100.0,
            "banrate": banrate / 100.0,
        }
    _validate_winrates({k: v["winrate"] for k, v in stats.items()})
    _validate_rates(stats)
    return stats


def load_winrates_by_role(path: str) -> dict[str, dict[str, dict[str, float]]]:
    """Load champion meta stats keyed by role.

    The CSV is expected to contain: patch, champion, role, winrate, pickrate, banrate.
    """
    df = pd.read_csv(path)
    required = {"patch", "champion", "role", "winrate", "pickrate", "banrate"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in winrate CSV: {sorted(missing)}")

    patch_values = df["patch"].dropna().unique()
    if len(patch_values) != 1:
        raise ValueError(f"Expected exactly one patch value, found: {patch_values}")
    patch = str(patch_values[0])
    _validate_patch(patch)

    stats: dict[str, dict[str, dict[str, float]]] = {}
    grouped = df.groupby(["champion", "role"])
    for (champion, role), group in grouped:
        winrate = float(group["winrate"].astype(float).max())
        pickrate = float(group["pickrate"].astype(float).max())
        banrate = float(group["banrate"].astype(float).max())
        champ_key = str(champion)
        role_key = str(role).upper()
        # All values in CSV are percentages, convert to decimals
        # winrate: 54.94 -> 0.5494, pickrate: 0.60 -> 0.006, banrate: 17.70 -> 0.177
        stats.setdefault(champ_key, {})[role_key] = {
            "winrate": winrate / 100.0,
            "pickrate": pickrate / 100.0,
            "banrate": banrate / 100.0,
        }
    for champ, roles in stats.items():
        for role, values in roles.items():
            pickrate = float(values.get("pickrate", 0.0))
            banrate = float(values.get("banrate", 0.0))
            if not 0.0 <= pickrate <= 1.0:
                raise ValueError(f"Pickrate out of bounds for {champ} {role}: {pickrate}")
            if not 0.0 <= banrate <= 1.0:
                raise ValueError(f"Banrate out of bounds for {champ} {role}: {banrate}")
    return stats


def _validate_patch(patch: str) -> None:
    """Ensure the winrate patch matches the current model patch."""
    from os import getenv

    expected = getenv("MODEL_PATCH_END")
    if not expected:
        return
    if patch != expected:
        raise ValueError(f"Winrate patch {patch} does not match model patch {expected}.")


def _validate_winrates(winrates: dict) -> None:
    """Validate champion names and winrate ranges."""
    for champion, rate in winrates.items():
        if not isinstance(champion, str) or not champion.strip():
            raise ValueError("Invalid champion name in winrate data.")
        value = float(rate)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"Winrate out of bounds for {champion}: {value}")


def _validate_rates(stats: dict[str, dict[str, float]]) -> None:
    """Validate pickrate/banrate ranges."""
    for champion, values in stats.items():
        pickrate = float(values.get("pickrate", 0.0))
        banrate = float(values.get("banrate", 0.0))
        if not 0.0 <= pickrate <= 1.0:
            raise ValueError(f"Pickrate out of bounds for {champion}: {pickrate}")
        if not 0.0 <= banrate <= 1.0:
            raise ValueError(f"Banrate out of bounds for {champion}: {banrate}")
