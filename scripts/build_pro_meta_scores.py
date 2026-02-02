"""Build pro meta scores from gol_gg_pro_meta.csv.

This script:
1. Calculates pro_meta_score from gol_gg_pro_meta.csv (role-agnostic)
2. Saves to pro_meta_scores.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from core.config import BAN_RATE_TOTAL_GAMES

TOTAL_PRO_GAMES = BAN_RATE_TOTAL_GAMES


def calculate_pro_meta_score(winrate: float, bans: int, total_games: int) -> float:
    """Calculate pro meta score for a champion.
    
    Formula:
    - pro_base: (pro_winrate - 45) / 10
    - pro_ban_bonus: (pro_banrate - 5) / 10 (where banrate = bans / 201)
    - pro_meta_score = min(1.0, (pro_base * 0.75) + (pro_ban_bonus * 0.25))
    - Set to 0 if total_games < 4
    
    Args:
        winrate: Winrate percentage (0-100)
        bans: Number of bans
        total_games: Total games (picks + bans)
    
    Returns:
        Pro meta score clamped to [0, 1]
    """
    # Set champions with < 4 total_games to 0
    if total_games < 4:
        return 0.0
    
    # Calculate pro_base: (winrate - 45) / 10
    pro_base = (winrate - 45.0) / 10.0
    pro_base = max(0.0, pro_base)  # Clamp to non-negative
    
    # Calculate pro_ban_bonus: (banrate - 5) / 10
    # banrate = bans / 201 (as percentage, so multiply by 100)
    banrate_pct = (bans / TOTAL_PRO_GAMES) * 100.0
    if banrate_pct <= 5.0:
        pro_ban_bonus = 0.0
    else:
        pro_ban_bonus = (banrate_pct - 5.0) / 10.0
        # Not clamped - can exceed 1.0
    
    # Calculate pro_meta_score
    pro_meta_score = (pro_base * 0.75) + (pro_ban_bonus * 0.25)
    pro_meta_score = min(1.0, max(0.0, pro_meta_score))  # Clamp to [0, 1]
    
    return pro_meta_score


def build_pro_meta_scores(pro_meta_path: Path, output_path: Path) -> None:
    """Build pro_meta_scores.csv from gol_gg_pro_meta.csv.
    
    Saves role-agnostic pro meta scores.
    """
    if not pro_meta_path.exists():
        print(f"Warning: {pro_meta_path} not found. Creating empty file.")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=["champion", "pro_meta_score"]).to_csv(output_path, index=False)
        return
    
    df = pd.read_csv(pro_meta_path)
    
    pro_scores = []
    for _, row in df.iterrows():
        champion = str(row["champion"]).strip()
        winrate = float(row.get("winrate", 0.0))
        bans = int(row.get("bans", 0))
        total_games = int(row.get("total_games", 0))
        
        pro_score = calculate_pro_meta_score(winrate, bans, total_games)
        pro_scores.append({
            "champion": champion,
            "pro_meta_score": pro_score,
        })
    
    # Save to CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df = pd.DataFrame(pro_scores)
    output_df = output_df.sort_values("champion")
    output_df.to_csv(output_path, index=False)
    print(f"Created {output_path} with {len(output_df)} entries")
    print(f"Sample entries:")
    print(output_df.head(10).to_string())


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Build pro meta scores from gol_gg_pro_meta.csv"
    )
    parser.add_argument(
        "--pro-meta",
        type=str,
        default="data/meta/gol_gg_pro_meta.csv",
        help="Path to pro meta data CSV",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/meta/pro_meta_scores.csv",
        help="Path to output pro_meta_scores.csv",
    )
    args = parser.parse_args()
    
    build_pro_meta_scores(Path(args.pro_meta), Path(args.output))


if __name__ == "__main__":
    main()
