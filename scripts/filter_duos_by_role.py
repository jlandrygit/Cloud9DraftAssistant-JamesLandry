"""Filter u_gg_duos.csv to only keep role-specific synergy pairs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

# Role synergy mappings: role -> expected duo_role
ROLE_SYNERGY_MAP = {
    "TOP": "JUNGLE",
    "JUNGLE": "SUPPORT",
    "MID": "JUNGLE",
    "ADC": "SUPPORT",
    "SUPPORT": "ADC",
}


def filter_duos_by_role(input_path: Path, output_path: Path) -> None:
    """Filter duos CSV to only keep rows matching role synergy pairs."""
    df = pd.read_csv(input_path)
    
    # Filter rows where duo_role matches the expected synergy role
    filtered_rows = []
    for _, row in df.iterrows():
        role = str(row["role"]).strip().upper()
        duo_role = str(row["duo_role"]).strip().upper()
        
        expected_duo_role = ROLE_SYNERGY_MAP.get(role)
        if expected_duo_role and duo_role == expected_duo_role:
            filtered_rows.append(row)
        elif role not in ROLE_SYNERGY_MAP:
            # Keep rows for roles not in the map (in case there are other roles)
            filtered_rows.append(row)
    
    filtered_df = pd.DataFrame(filtered_rows)
    
    # Write back to the same file (or output path if specified)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    filtered_df.to_csv(output_path, index=False)
    
    print(f"Filtered {len(df)} rows to {len(filtered_df)} rows")
    print(f"Removed {len(df) - len(filtered_df)} rows that didn't match role synergy pairs")


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter u_gg_duos.csv by role synergy pairs")
    parser.add_argument(
        "--input",
        type=Path,
        default="data/meta/u_gg_duos.csv",
        help="Input CSV path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default="data/meta/u_gg_duos.csv",
        help="Output CSV path (defaults to input path)",
    )
    args = parser.parse_args()
    
    filter_duos_by_role(args.input, args.output)


if __name__ == "__main__":
    main()
