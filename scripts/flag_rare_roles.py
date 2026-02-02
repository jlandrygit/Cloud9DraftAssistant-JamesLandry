"""Flag champion-role pairs that are not seen in recent pro play."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare U.GG roles to pro-play stats and flag unseen roles."
    )
    parser.add_argument(
        "--roles",
        default="data/meta/u_gg_roles.csv",
        help="Path to u_gg_roles.csv",
    )
    parser.add_argument(
        "--stats",
        default="data/processed/player_champion_stats.csv",
        help="Path to player_champion_stats.csv",
    )
    parser.add_argument(
        "--output",
        default="data/processed/roles_unseen_in_pro.csv",
        help="Path to write unseen champion-role pairs",
    )
    args = parser.parse_args()

    roles_path = Path(args.roles)
    stats_path = Path(args.stats)
    output_path = Path(args.output)

    if not roles_path.exists():
        raise FileNotFoundError(f"Missing roles file: {roles_path}")
    if not stats_path.exists():
        raise FileNotFoundError(f"Missing stats file: {stats_path}")

    roles_df = pd.read_csv(roles_path)
    stats_df = pd.read_csv(stats_path)

    required_roles_cols = {"champion", "role"}
    missing_roles = required_roles_cols - set(roles_df.columns)
    if missing_roles:
        raise ValueError(f"u_gg_roles.csv missing columns: {sorted(missing_roles)}")

    required_stats_cols = {"champion_name", "role"}
    missing_stats = required_stats_cols - set(stats_df.columns)
    if missing_stats:
        raise ValueError(
            f"player_champion_stats.csv missing columns: {sorted(missing_stats)}"
        )

    roles_df = roles_df[["champion", "role"]].dropna().drop_duplicates()
    stats_df = stats_df[["champion_name", "role"]].dropna().drop_duplicates()

    roles_df["champion"] = roles_df["champion"].astype(str)
    roles_df["role"] = roles_df["role"].astype(str)
    stats_df["champion_name"] = stats_df["champion_name"].astype(str)
    stats_df["role"] = stats_df["role"].astype(str)

    seen_pairs = set(zip(stats_df["champion_name"], stats_df["role"]))
    unseen_rows = [
        {"champion": champ, "role": role}
        for champ, role in zip(roles_df["champion"], roles_df["role"])
        if (champ, role) not in seen_pairs
    ]

    unseen_df = pd.DataFrame(unseen_rows).sort_values(["champion", "role"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    unseen_df.to_csv(output_path, index=False)

    print(f"Unseen champion-role pairs: {len(unseen_df)}")
    print(f"Wrote: {output_path}")


if __name__ == "__main__":
    main()
