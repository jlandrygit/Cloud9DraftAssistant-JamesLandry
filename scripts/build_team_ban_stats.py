"""Compile ban data against each team from LoLDraftingData.xlsx.

This script analyzes which champions are banned against each team to identify:
1. Meta champions that are commonly banned (strong on patch)
2. Team-specific bans (champions teams are known for playing)

The output will be used to calculate opponent-based comfort scores.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

# Add parent directory to path for imports
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.load_drafts_excel import load_draft_excel
from draft.draft_order import DRAFT_SEQUENCE


def extract_ban_columns() -> tuple[list[str], list[str]]:
    """Extract blue ban and red ban column names from DRAFT_SEQUENCE.
    
    Returns:
        Tuple of (blue_ban_columns, red_ban_columns)
    """
    blue_bans = []
    red_bans = []
    
    for step in DRAFT_SEQUENCE:
        _, side, action_type, column_name = step
        if action_type == "BAN":
            if side == "BLUE":
                blue_bans.append(column_name)
            else:
                red_bans.append(column_name)
    
    return blue_bans, red_bans


def compile_team_ban_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compile ban statistics for each team.
    
    For each team, tracks:
    - How many times each champion was banned against them
    - Total number of games the team played
    - Ban rate for each champion against that team
    
    Args:
        df: DataFrame from LoLDraftingData.xlsx with draft data
    
    Returns:
        DataFrame with columns: team_name, champion_name, bans_against, total_games, ban_rate
    """
    blue_ban_cols, red_ban_cols = extract_ban_columns()
    
    # Track bans against each team
    # Structure: {team_name: {champion: count}}
    team_bans: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    team_game_count: dict[str, int] = defaultdict(int)
    
    for _, row in df.iterrows():
        blue_team = str(row.get("BLUE SIDE", "")).strip()
        red_team = str(row.get("RED SIDE", "")).strip()
        
        if not blue_team or not red_team:
            continue
        
        # Count games for each team
        team_game_count[blue_team] += 1
        team_game_count[red_team] += 1
        
        # Extract bans made by each side (bans against the opponent)
        # Blue side bans are bans against red team
        # Red side bans are bans against blue team
        for ban_col in blue_ban_cols:
            ban_champion = str(row.get(ban_col, "")).strip()
            if ban_champion and ban_champion.upper() not in {"B", "R", ""}:
                team_bans[red_team][ban_champion] += 1
        
        for ban_col in red_ban_cols:
            ban_champion = str(row.get(ban_col, "")).strip()
            if ban_champion and ban_champion.upper() not in {"B", "R", ""}:
                team_bans[blue_team][ban_champion] += 1
    
    # Convert to DataFrame
    records = []
    for team_name, bans_dict in team_bans.items():
        total_games = team_game_count.get(team_name, 0)
        if total_games == 0:
            continue
        
        for champion, bans_against in bans_dict.items():
            ban_rate = bans_against / total_games if total_games > 0 else 0.0
            records.append({
                "team_name": team_name,
                "champion_name": champion,
                "bans_against": bans_against,
                "total_games": total_games,
                "ban_rate": ban_rate,
            })
    
    result_df = pd.DataFrame(records)
    
    # Sort by team_name, then by ban_rate descending
    if not result_df.empty:
        result_df = result_df.sort_values(
            ["team_name", "ban_rate", "bans_against"],
            ascending=[True, False, False]
        )
    
    return result_df


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Compile ban statistics against each team from LoLDraftingData.xlsx"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="LoLDraftingData.xlsx",
        help="Path to LoLDraftingData.xlsx",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/team_ban_stats.csv",
        help="Output CSV path for team ban statistics",
    )
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)
    
    print(f"Loading draft data from {input_path}...")
    try:
        df = load_draft_excel(str(input_path))
        print(f"Loaded {len(df)} draft rows")
    except Exception as e:
        print(f"Error loading draft data: {e}")
        sys.exit(1)
    
    print("Compiling ban statistics per team...")
    ban_stats_df = compile_team_ban_stats(df)
    
    if ban_stats_df.empty:
        print("Warning: No ban statistics compiled. Check input data format.")
        sys.exit(1)
    
    print(f"Compiled ban statistics for {ban_stats_df['team_name'].nunique()} teams")
    print(f"Total team-champion pairs: {len(ban_stats_df)}")
    
    # Show summary statistics
    print("\nSummary:")
    print(f"  Teams with ban data: {ban_stats_df['team_name'].nunique()}")
    print(f"  Unique champions banned: {ban_stats_df['champion_name'].nunique()}")
    print(f"  Average bans per team: {ban_stats_df.groupby('team_name')['bans_against'].sum().mean():.1f}")
    
    # Show top banned champions overall
    top_banned = ban_stats_df.groupby("champion_name")["bans_against"].sum().sort_values(ascending=False).head(10)
    print("\nTop 10 most banned champions (across all teams):")
    for champion, total_bans in top_banned.items():
        print(f"  {champion}: {total_bans} total bans")
    
    # Save to CSV
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ban_stats_df.to_csv(output_path, index=False)
    print(f"\nSaved ban statistics to {output_path}")
    
    # Show sample of output
    print("\nSample output (first 10 rows):")
    print(ban_stats_df.head(10).to_string())


if __name__ == "__main__":
    main()
