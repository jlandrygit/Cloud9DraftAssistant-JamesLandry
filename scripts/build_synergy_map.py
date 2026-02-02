"""Build a simple synergy map from historical series state data."""

from __future__ import annotations

import sys
import argparse
import ast
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from data.meta.load_u_gg_roles import load_roles

ADC_SUPPORT_BONUS = 0.5
MID_JUNGLE_BONUS = 0.4
TOP_PENALTY_MULTIPLIER = 0.5


def _pair_weight(champ_a: str, champ_b: str, roles_map: dict[str, set[str]]) -> float:
    """Return a non-negative weight for a champion pair."""
    base = 1.0
    roles_a = roles_map.get(champ_a, set())
    roles_b = roles_map.get(champ_b, set())

    bonus = 0.0
    if ("ADC" in roles_a and "SUPPORT" in roles_b) or (
        "SUPPORT" in roles_a and "ADC" in roles_b
    ):
        bonus += ADC_SUPPORT_BONUS
    if ("MID" in roles_a and "JUNGLE" in roles_b) or (
        "JUNGLE" in roles_a and "MID" in roles_b
    ):
        bonus += MID_JUNGLE_BONUS

    weight = base + bonus
    if "TOP" in roles_a or "TOP" in roles_b:
        weight *= TOP_PENALTY_MULTIPLIER
    return max(0.0, weight)


def build_synergy_map(
    series_path: Path, roles_path: Path, output_path: Path
) -> dict[str, dict[str, float]]:
    """Build a champion synergy map from series state data."""
    roles_map = load_roles(str(roles_path))
    champion_list = _load_champion_list()
    pair_counts: dict[tuple[str, str], float] = defaultdict(float)

    series = json.loads(series_path.read_text(encoding="utf-8"))
    for entry in series:
        games = entry.get("games", [])
        for game in games:
            for team in game.get("teams", []):
                players = team.get("players", [])
                champs = []
                for player in players:
                    character = player.get("character") or {}
                    champ = character.get("name")
                    if champ:
                        champs.append(str(champ))
                unique_champs = sorted(set(champs))
                if len(unique_champs) < 2:
                    continue
                for champ_a, champ_b in combinations(unique_champs, 2):
                    weight = _pair_weight(champ_a, champ_b, roles_map)
                    pair_counts[(champ_a, champ_b)] += weight

    synergy: dict[str, dict[str, float]] = defaultdict(dict)
    for (champ_a, champ_b), weight in pair_counts.items():
        synergy[champ_a][champ_b] = synergy[champ_a].get(champ_b, 0.0) + weight
        synergy[champ_b][champ_a] = synergy[champ_b].get(champ_a, 0.0) + weight

    # Normalize per champion so scores are in [0, 1].
    normalized: dict[str, dict[str, float]] = {}
    for champ, partners in synergy.items():
        max_value = max(partners.values()) if partners else 0.0
        if max_value <= 0.0:
            continue
        normalized[champ] = {
            partner: round(value / max_value, 6) for partner, value in partners.items()
        }

    for champ in champion_list:
        normalized.setdefault(champ, {})

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(normalized, indent=2), encoding="utf-8")
    return normalized


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a synergy map from series data.")
    parser.add_argument(
        "--series",
        default="data/ingestion/series_states.json",
        help="Path to series_states.json",
    )
    parser.add_argument(
        "--roles",
        default="data/meta/u_gg_roles.csv",
        help="Path to champion roles CSV",
    )
    parser.add_argument(
        "--output",
        default="data/processed/synergy_map.json",
        help="Path to write synergy_map.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_synergy_map(Path(args.series), Path(args.roles), Path(args.output))
    print(f"Wrote synergy map to {args.output}")


def _load_champion_list() -> list[str]:
    """Load the CHAMPIONS list from app.py without importing Streamlit."""
    app_path = PROJECT_ROOT / "app.py"
    if not app_path.exists():
        return []
    text = app_path.read_text(encoding="utf-8")
    marker = "CHAMPIONS = ["
    start = text.find(marker)
    if start == -1:
        return []
    start = text.find("[", start)
    if start == -1:
        return []
    bracket_depth = 0
    end = None
    for idx in range(start, len(text)):
        char = text[idx]
        if char == "[":
            bracket_depth += 1
        elif char == "]":
            bracket_depth -= 1
            if bracket_depth == 0:
                end = idx + 1
                break
    if end is None:
        return []
    snippet = text[start:end]
    try:
        parsed = ast.literal_eval(snippet)
    except Exception:
        return []
    if not isinstance(parsed, list):
        return []
    return [str(item) for item in parsed if isinstance(item, str)]


if __name__ == "__main__":
    main()
