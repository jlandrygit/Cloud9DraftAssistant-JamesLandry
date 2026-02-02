"""Load and validate U.GG champion role mappings."""

from __future__ import annotations

from collections import defaultdict

import pandas as pd


def load_roles(path: str) -> dict[str, set[str]]:
    """Return champion -> set of roles from a U.GG roles CSV."""
    df = pd.read_csv(path)
    required = {"champion", "role"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in roles CSV: {sorted(missing)}")

    roles_map: dict[str, set[str]] = defaultdict(set)
    for _, row in df[["champion", "role"]].dropna().iterrows():
        champion = str(row["champion"]).strip()
        role = str(row["role"]).strip().upper()
        if not champion or not role:
            continue
        roles_map[champion].add(role)
    return dict(roles_map)
