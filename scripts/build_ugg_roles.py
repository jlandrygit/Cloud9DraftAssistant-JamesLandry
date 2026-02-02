"""Build a champion-to-roles CSV from U.GG winrate data.

This script reads `data/meta/u_gg_winrates.csv` and outputs a flat mapping of
champion -> role based on the roles present in the U.GG data. The output is
used to prevent recommendations that conflict with already-filled roles.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = {"champion", "role"}


def build_roles_csv(input_path: Path, output_path: Path) -> pd.DataFrame:
    """Return a de-duplicated champion-role table and write it to disk."""
    df = pd.read_csv(input_path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    roles = (
        df[["champion", "role"]]
        .dropna()
        .drop_duplicates()
        .sort_values(["champion", "role"], kind="mergesort")
        .reset_index(drop=True)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    roles.to_csv(output_path, index=False)
    return roles


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build U.GG champion roles CSV.")
    parser.add_argument(
        "--input",
        default="data/meta/u_gg_winrates.csv",
        help="Path to the U.GG winrates CSV.",
    )
    parser.add_argument(
        "--output",
        default="data/meta/u_gg_roles.csv",
        help="Path to write the champion roles CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    roles = build_roles_csv(Path(args.input), Path(args.output))
    print(f"Wrote {len(roles)} rows to {args.output}")


if __name__ == "__main__":
    main()
