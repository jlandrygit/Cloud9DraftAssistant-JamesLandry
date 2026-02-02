"""Replace UNKNOWN roles using roster_map.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

DEFAULT_STATS_PATH = "data/processed/player_champion_stats.csv"
DEFAULT_ROSTER_PATH = "data/rosters/roster_map.json"


def _load_player_role_map(roster_path: Path) -> dict[str, str]:
    """Build a player -> role mapping from the roster map."""
    data = json.loads(roster_path.read_text(encoding="utf-8"))
    mapping: dict[str, str] = {}
    for team, roles in data.items():
        if not isinstance(roles, dict):
            continue
        for role, player in roles.items():
            if not player or not str(player).strip():
                continue
            mapping[str(player)] = str(role).upper()
    return mapping


def fix_roles(stats_path: Path, roster_path: Path, output_path: Path) -> None:
    """Rewrite UNKNOWN roles using roster data."""
    df = pd.read_csv(stats_path)
    required = {"player_name", "role"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    player_roles = _load_player_role_map(roster_path)
    if not player_roles:
        raise ValueError("Roster map is empty or invalid.")

    unknown_mask = df["role"].astype(str).str.upper().eq("UNKNOWN")
    if unknown_mask.any():
        df.loc[unknown_mask, "role"] = (
            df.loc[unknown_mask, "player_name"].map(player_roles).fillna("UNKNOWN")
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fix UNKNOWN roles using roster map.")
    parser.add_argument("--stats", default=DEFAULT_STATS_PATH, help="Input stats CSV.")
    parser.add_argument("--rosters", default=DEFAULT_ROSTER_PATH, help="Roster map JSON.")
    parser.add_argument(
        "--output",
        default=DEFAULT_STATS_PATH,
        help="Output path (defaults to overwriting input).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fix_roles(Path(args.stats), Path(args.rosters), Path(args.output))
    print(f"Wrote updated roles to {args.output}")


if __name__ == "__main__":
    main()
