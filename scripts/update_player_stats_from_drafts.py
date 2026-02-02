"""Update player_champion_stats.csv from LoLDraftingData.xlsx.

For each draft in LoLDraftingData.xlsx:
1. Identifies teams (blue side and red side)
2. For each pick, determines:
   - Which team made the pick
   - Champion name and role (from u_gg_roles.csv)
   - Player name (from roster_map.json)
3. Updates player_champion_stats.csv:
   - Increments games_played
   - Increments wins if that team won
   - Recalculates win_rate
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

# Add parent directory to path for imports
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.load_drafts_excel import load_draft_excel
from draft.draft_order import DRAFT_SEQUENCE


def load_roster_map(roster_path: Path) -> dict[str, dict[str, str]]:
    """Load roster map from JSON file.
    
    Returns:
        dict[team_name, dict[role, player_name]]
    """
    with open(roster_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_role_map(roles_path: Path) -> dict[str, list[str]]:
    """Load champion roles from CSV.
    
    Returns:
        dict[champion, list[roles]]
    """
    df = pd.read_csv(roles_path)
    role_map: dict[str, list[str]] = {}
    for _, row in df.iterrows():
        champion = str(row["champion"]).strip()
        role = str(row["role"]).strip().upper()
        if champion not in role_map:
            role_map[champion] = []
        role_map[champion].append(role)
    return role_map


def load_player_stats(stats_path: Path) -> pd.DataFrame:
    """Load or create player_champion_stats.csv.
    
    Returns:
        DataFrame with columns: player_name, champion_name, role, games_played, wins, win_rate
    """
    if stats_path.exists():
        return pd.read_csv(stats_path)
    else:
        # Create empty DataFrame
        return pd.DataFrame(columns=["player_name", "champion_name", "role", "games_played", "wins", "win_rate"])


def get_pick_team_and_role(column_name: str, draft_sequence: list) -> tuple[str, str] | None:
    """Determine which team and role a pick column belongs to.
    
    Args:
        column_name: Column name like "BP1", "RP1", etc.
        draft_sequence: The DRAFT_SEQUENCE list
    
    Returns:
        Tuple of (side, role) or None if not found
    """
    for step in draft_sequence:
        step_idx, side, action_type, col_name = step
        if col_name == column_name and action_type == "PICK":
            # Determine role based on pick order
            # BP1 = first blue pick = TOP (if blue picks first)
            # RP1 = first red pick = TOP (if red picks first)
            # But we need to map based on actual draft order
            # Let's use the step index to determine role
            pick_order = {
                7: "TOP",    # BP1 - first pick
                8: "TOP",    # RP1 - first pick
                9: "JUNGLE", # RP2 - second pick
                10: "JUNGLE", # BP2 - second pick
                11: "MID",   # BP3 - third pick
                12: "MID",   # RP3 - third pick
                17: "ADC",   # RP4 - fourth pick
                18: "ADC",   # BP4 - fourth pick
                19: "SUPPORT", # BP5 - fifth pick
                20: "SUPPORT", # RP5 - fifth pick
            }
            role = pick_order.get(step_idx)
            if role:
                return (side, role)
    return None


def determine_pick_role(column_name: str, side: str, step_index: int) -> str:
    """Determine the role for a pick based on draft order.
    
    Standard draft order:
    - First pick phase: TOP, JUNGLE, MID (alternating)
    - Second pick phase: ADC, SUPPORT (alternating)
    """
    # Map step indices to roles
    role_map = {
        7: "TOP",      # BP1
        8: "TOP",      # RP1
        9: "JUNGLE",   # RP2
        10: "JUNGLE",  # BP2
        11: "MID",     # BP3
        12: "MID",     # RP3
        17: "ADC",     # RP4
        18: "ADC",     # BP4
        19: "SUPPORT", # BP5
        20: "SUPPORT", # RP5
    }
    return role_map.get(step_index, "UNKNOWN")


def update_player_stats_from_draft(
    draft_row: pd.Series,
    roster_map: dict[str, dict[str, str]],
    role_map: dict[str, list[str]],
    player_stats: pd.DataFrame,
) -> pd.DataFrame:
    """Update player stats for a single draft.
    
    Args:
        draft_row: Row from LoLDraftingData.xlsx
        roster_map: Team roster mapping
        role_map: Champion to roles mapping
        player_stats: Current player stats DataFrame
    
    Returns:
        Updated player_stats DataFrame
    """
    # Get teams
    blue_team = str(draft_row.get("BLUE SIDE", "")).strip()
    red_team = str(draft_row.get("RED SIDE", "")).strip()
    
    # Get winner (should always be present now)
    winner = None
    if "WINNER" in draft_row:
        winner_str = str(draft_row.get("WINNER", "")).strip()
        if winner_str and winner_str != "":
            winner = winner_str
    
    # Process each pick
    pick_columns = ["BP1", "RP1", "RP2", "BP2", "BP3", "RP3", "RP4", "BP4", "BP5", "RP5"]
    
    for col in pick_columns:
        if col not in draft_row:
            continue
        
        champion = str(draft_row[col]).strip()
        if not champion or champion == "" or champion.upper() in ["B", "R"]:
            continue
        
        # Determine which team made this pick
        if col.startswith("B"):
            team = blue_team
        else:
            team = red_team
        
        # Get team roster
        if team not in roster_map:
            continue
        
        team_roster = roster_map[team]
        
        # Get champion's possible roles
        champion_roles = role_map.get(champion, [])
        if not champion_roles:
            # Champion not in role map, skip
            continue
        
        # Determine if this team won
        won = False
        if winner:
            won = (winner == team)
        
        # Update stats for ALL players who could play this champion
        # This handles flex picks by giving benefit of the doubt to all possible players
        matched = False
        for role, player_name in team_roster.items():
            # Check if this champion can play this role
            if role in champion_roles:
                # This player could have played this champion in this role
                matched = True
                
                # Update player stats
                # Find existing entry or create new
                mask = (
                    (player_stats["player_name"] == player_name) &
                    (player_stats["champion_name"] == champion) &
                    (player_stats["role"] == role)
                )
                
                if mask.any():
                    # Update existing entry
                    idx = player_stats[mask].index[0]
                    player_stats.at[idx, "games_played"] += 1
                    if won:
                        player_stats.at[idx, "wins"] += 1
                    # Recalculate win_rate
                    games = player_stats.at[idx, "games_played"]
                    wins = player_stats.at[idx, "wins"]
                    player_stats.at[idx, "win_rate"] = wins / games if games > 0 else 0.0
                else:
                    # Create new entry
                    new_row = {
                        "player_name": player_name,
                        "champion_name": champion,
                        "role": role,
                        "games_played": 1,
                        "wins": 1 if won else 0,
                        "win_rate": 1.0 if won else 0.0,
                    }
                    player_stats = pd.concat([player_stats, pd.DataFrame([new_row])], ignore_index=True)
        
        # If no match found, skip this pick (champion doesn't match any player's role)
        if not matched:
            continue
    
    return player_stats


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Update player_champion_stats.csv from LoLDraftingData.xlsx"
    )
    parser.add_argument(
        "--drafts",
        type=str,
        default="LoLDraftingData.xlsx",
        help="Path to LoLDraftingData.xlsx",
    )
    parser.add_argument(
        "--roster",
        type=str,
        default="data/rosters/roster_map.json",
        help="Path to roster_map.json",
    )
    parser.add_argument(
        "--roles",
        type=str,
        default="data/meta/u_gg_roles.csv",
        help="Path to u_gg_roles.csv",
    )
    parser.add_argument(
        "--stats",
        type=str,
        default="data/processed/player_champion_stats.csv",
        help="Path to player_champion_stats.csv",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path (defaults to --stats path)",
    )
    args = parser.parse_args()
    
    drafts_path = Path(args.drafts)
    roster_path = Path(args.roster)
    roles_path = Path(args.roles)
    stats_path = Path(args.stats)
    output_path = Path(args.output) if args.output else stats_path
    
    print("Loading data...")
    drafts_df = load_draft_excel(str(drafts_path))
    print(f"  Loaded {len(drafts_df)} drafts")
    
    roster_map = load_roster_map(roster_path)
    print(f"  Loaded {len(roster_map)} team rosters")
    
    role_map = load_role_map(roles_path)
    print(f"  Loaded roles for {len(role_map)} champions")
    
    player_stats = load_player_stats(stats_path)
    print(f"  Loaded {len(player_stats)} existing player/champion/role entries")
    
    print("\nProcessing drafts...")
    updated_count = 0
    new_count = 0
    
    for idx, row in drafts_df.iterrows():
        if idx % 100 == 0:
            print(f"  Processing draft {idx + 1}/{len(drafts_df)}...")
        
        initial_count = len(player_stats)
        player_stats = update_player_stats_from_draft(row, roster_map, role_map, player_stats)
        
        if len(player_stats) > initial_count:
            new_count += len(player_stats) - initial_count
        else:
            updated_count += 1
    
    print(f"\nUpdated {updated_count} existing entries")
    print(f"Created {new_count} new entries")
    print(f"Total entries: {len(player_stats)}")
    
    # Sort and save
    player_stats = player_stats.sort_values(["player_name", "champion_name", "role"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    player_stats.to_csv(output_path, index=False)
    
    print(f"\nSaved updated stats to {output_path}")


if __name__ == "__main__":
    main()
