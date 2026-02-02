"""Filter u_gg_counters.csv to only keep rows where counter_champion plays the specified role."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def filter_counters_by_role(
    counters_path: Path, roles_path: Path, output_path: Path
) -> None:
    """Filter counters CSV to only keep rows where counter_champion plays the role."""
    # Load roles CSV to get valid champion-role pairs
    roles_df = pd.read_csv(roles_path)
    
    # Create a set of valid (champion, role) pairs for fast lookup
    valid_pairs = set()
    for _, row in roles_df.iterrows():
        champion = str(row["champion"]).strip()
        role = str(row["role"]).strip().upper()
        valid_pairs.add((champion, role))
    
    # Load counters CSV
    counters_df = pd.read_csv(counters_path)
    
    # Filter rows where counter_champion has the role
    filtered_rows = []
    removed_count = 0
    
    for _, row in counters_df.iterrows():
        counter_champion = str(row["counter_champion"]).strip()
        role = str(row["role"]).strip().upper()
        
        # Check if this counter_champion plays this role
        if (counter_champion, role) in valid_pairs:
            filtered_rows.append(row)
        else:
            removed_count += 1
    
    filtered_df = pd.DataFrame(filtered_rows)
    
    # Write back to the same file (or output path if specified)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    filtered_df.to_csv(output_path, index=False)
    
    print(f"Filtered {len(counters_df)} rows to {len(filtered_df)} rows")
    print(f"Removed {removed_count} rows where counter_champion doesn't play the role")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter u_gg_counters.csv by valid champion-role pairs"
    )
    parser.add_argument(
        "--counters",
        type=Path,
        default="data/meta/u_gg_counters.csv",
        help="Input counters CSV path",
    )
    parser.add_argument(
        "--roles",
        type=Path,
        default="data/meta/u_gg_roles.csv",
        help="Roles CSV path for validation",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default="data/meta/u_gg_counters.csv",
        help="Output CSV path (defaults to input path)",
    )
    args = parser.parse_args()
    
    filter_counters_by_role(args.counters, args.roles, args.output)


if __name__ == "__main__":
    main()
