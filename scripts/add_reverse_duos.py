"""Add reverse duo entries for TOP->JUNGLE, JUNGLE->SUPPORT, and MID->JUNGLE."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

# Role synergy mappings that need reverse entries
REVERSE_PAIRS = {
    ("TOP", "JUNGLE"): ("JUNGLE", "TOP"),
    ("JUNGLE", "SUPPORT"): ("SUPPORT", "JUNGLE"),
    ("MID", "JUNGLE"): ("JUNGLE", "MID"),
}


def add_reverse_duos(input_path: Path, output_path: Path) -> None:
    """Add reverse duo entries for specified role pairs."""
    df = pd.read_csv(input_path)
    
    # Create a set of existing entries to avoid duplicates
    existing_entries = set()
    for _, row in df.iterrows():
        champion = str(row["champion"]).strip()
        role = str(row["role"]).strip().upper()
        duo_champion = str(row["duo_champion"]).strip()
        duo_role = str(row["duo_role"]).strip().upper()
        existing_entries.add((champion, role, duo_champion, duo_role))
    
    # Find entries that need reverse entries and create them
    reverse_rows = []
    added_count = 0
    
    for _, row in df.iterrows():
        champion = str(row["champion"]).strip()
        role = str(row["role"]).strip().upper()
        duo_champion = str(row["duo_champion"]).strip()
        duo_role = str(row["duo_role"]).strip().upper()
        winrate = float(row["winrate"])
        if "matches" in row and pd.notna(row["matches"]):
            matches = int(row["matches"])
        else:
            matches = 0
        
        # Check if this entry matches a pattern that needs a reverse
        if (role, duo_role) in REVERSE_PAIRS:
            # Create reverse entry
            reverse_role, reverse_duo_role = REVERSE_PAIRS[(role, duo_role)]
            reverse_entry = (duo_champion, reverse_role, champion, reverse_duo_role)
            
            # Only add if it doesn't already exist
            if reverse_entry not in existing_entries:
                reverse_rows.append({
                    "champion": duo_champion,
                    "role": reverse_role,
                    "duo_champion": champion,
                    "duo_role": reverse_duo_role,
                    "winrate": winrate,
                    "matches": matches,
                })
                existing_entries.add(reverse_entry)
                added_count += 1
    
    # Combine original and reverse entries
    if reverse_rows:
        reverse_df = pd.DataFrame(reverse_rows)
        combined_df = pd.concat([df, reverse_df], ignore_index=True)
    else:
        combined_df = df
    
    # Write to output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(output_path, index=False)
    
    print(f"Added {added_count} reverse duo entries")
    print(f"Total entries: {len(df)} -> {len(combined_df)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add reverse duo entries for TOP->JUNGLE, JUNGLE->SUPPORT, MID->JUNGLE"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default="data/meta/u_gg_duos.csv",
        help="Input duos CSV path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default="data/meta/u_gg_duos.csv",
        help="Output duos CSV path (defaults to input path)",
    )
    args = parser.parse_args()
    
    add_reverse_duos(args.input, args.output)


if __name__ == "__main__":
    main()
