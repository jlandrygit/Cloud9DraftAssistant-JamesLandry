"""Build a counter map from historical series state data."""

from __future__ import annotations

import argparse
import ast
import json
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MIN_MATCHUPS = 5
WINRATE_THRESHOLD = 0.50


def build_counter_map(series_path: Path, output_path: Path) -> dict[str, dict[str, float]]:
    """Build a champion counter map from series state data."""
    series = json.loads(series_path.read_text(encoding="utf-8"))
    counts: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: {"wins": 0, "total": 0})

    for entry in series:
        winner = _series_winner(entry.get("teams", []))
        if not winner:
            continue
        for game in entry.get("games", []):
            teams = game.get("teams", [])
            if len(teams) != 2:
                continue
            team_a = teams[0]
            team_b = teams[1]
            champs_a = _extract_champions(team_a.get("players", []))
            champs_b = _extract_champions(team_b.get("players", []))
            if not champs_a or not champs_b:
                continue
            if team_a.get("name") == winner:
                winners, losers = champs_a, champs_b
            elif team_b.get("name") == winner:
                winners, losers = champs_b, champs_a
            else:
                continue
            for w in winners:
                for l in losers:
                    counts[(w, l)]["wins"] += 1
                    counts[(w, l)]["total"] += 1
                    counts[(l, w)]["total"] += 1

    counter_map: dict[str, dict[str, float]] = defaultdict(dict)
    for (champ, opp), data in counts.items():
        total = data["total"]
        if total < MIN_MATCHUPS:
            continue
        winrate = data["wins"] / total
        if winrate > WINRATE_THRESHOLD:
            counter_map[champ][opp] = round(winrate, 6)

    for champ in _load_champion_list():
        counter_map.setdefault(champ, {})

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(counter_map, indent=2), encoding="utf-8")
    return counter_map


def _series_winner(teams: list[dict]) -> str | None:
    for team in teams:
        if team.get("won") is True and team.get("name"):
            return str(team["name"])
    return None


def _extract_champions(players: list[dict]) -> list[str]:
    champs: list[str] = []
    for player in players:
        character = player.get("character") or {}
        champ = character.get("name")
        if champ:
            champs.append(str(champ))
    return list(set(champs))


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a counter map from series data.")
    parser.add_argument(
        "--series",
        default="data/ingestion/series_states.json",
        help="Path to series_states.json",
    )
    parser.add_argument(
        "--output",
        default="data/processed/counter_map.json",
        help="Path to write counter_map.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_counter_map(Path(args.series), Path(args.output))
    print(f"Wrote counter map to {args.output}")


if __name__ == "__main__":
    main()
