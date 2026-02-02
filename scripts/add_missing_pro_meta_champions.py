"""Add missing champions to gol_gg_pro_meta.csv with zero values."""

from __future__ import annotations

import argparse
import ast
from pathlib import Path

import pandas as pd


def normalize_champion_name(name: str) -> str:
    """Normalize champion name to match CSV format (remove spaces, apostrophes, periods, &)."""
    normalized = name.strip()
    normalized = normalized.replace(" ", "")
    normalized = normalized.replace("'", "")
    normalized = normalized.replace(".", "")
    normalized = normalized.replace("&", "")
    return normalized


def load_champions_from_app(app_path: Path) -> list[str]:
    """Load CHAMPIONS list from app.py."""
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


def add_missing_champions(
    csv_path: Path,
    app_path: Path,
    output_path: Path | None = None,
) -> None:
    """Add missing champions to gol_gg_pro_meta.csv with zero values."""
    if output_path is None:
        output_path = csv_path
    
    # Load existing CSV
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        existing_normalized = {normalize_champion_name(champ) for champ in df["champion"]}
    else:
        df = pd.DataFrame(columns=["champion", "picks", "bans", "total_games", "wins", "winrate"])
        existing_normalized = set()
    
    # Load all champions from app.py
    all_champions = load_champions_from_app(app_path)
    if not all_champions:
        print("Warning: Could not load champions from app.py")
        return
    
    # Find missing champions
    missing_champions = []
    for champ in all_champions:
        normalized = normalize_champion_name(champ)
        if normalized not in existing_normalized:
            missing_champions.append(normalized)
    
    if not missing_champions:
        print("No missing champions found. All champions are already in the CSV.")
        return
    
    # Add missing champions with zero values
    new_rows = []
    for champ in sorted(missing_champions):
        new_rows.append({
            "champion": champ,
            "picks": 0,
            "bans": 0,
            "total_games": 0,
            "wins": 0.0,
            "winrate": 0.0,
        })
    
    # Append new rows to DataFrame
    new_df = pd.DataFrame(new_rows)
    combined_df = pd.concat([df, new_df], ignore_index=True)
    
    # Sort by champion name
    combined_df = combined_df.sort_values("champion")
    
    # Save to CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(output_path, index=False)
    
    print(f"Added {len(missing_champions)} missing champions:")
    for champ in sorted(missing_champions):
        print(f"  - {champ}")
    print(f"\nTotal champions in CSV: {len(df)} -> {len(combined_df)}")
    print(f"Saved to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add missing champions to gol_gg_pro_meta.csv"
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default="data/meta/gol_gg_pro_meta.csv",
        help="Path to gol_gg_pro_meta.csv",
    )
    parser.add_argument(
        "--app",
        type=Path,
        default="app.py",
        help="Path to app.py",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (defaults to --csv path)",
    )
    args = parser.parse_args()
    
    add_missing_champions(args.csv, args.app, args.output)


if __name__ == "__main__":
    main()
