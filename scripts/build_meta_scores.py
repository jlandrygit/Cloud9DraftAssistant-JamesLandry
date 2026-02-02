"""Pre-calculate meta scores for all champion/role combinations."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

META_WIN_WEIGHT = 0.75
META_BAN_WEIGHT = 0.25


def _pickrate_penalty(pickrate: float) -> float:
    """Penalize only very low pick rates (<1%).
    
    Assumes pickrate is in decimal format (0.01 = 1%).
    """
    if pickrate >= 0.01:
        return 0.0
    return ((0.01 - pickrate) / 0.01)


def _calculate_ban_bonus(banrate: float) -> float:
    """Calculate ban bonus: (banrate - 5) / 10.
    
    Only rewards champions with >5% banrate.
    Assumes banrate is in decimal format (0.05 = 5%).
    Not clamped - can exceed 1.0 for very high banrates.
    """
    if banrate <= 0.05:  # 5% in decimal
        return 0.0
    bonus = (banrate - 0.05) / 0.10  # (banrate - 5%) / 10%
    return bonus


def calculate_meta_score(winrate: float, pickrate: float, banrate: float) -> float:
    """Calculate meta score for a champion/role combination.
    
    Args:
        winrate: Winrate in decimal format (0.5494 = 54.94%)
        pickrate: Pickrate in decimal format (0.043 = 4.3%)
        banrate: Banrate in decimal format (0.29 = 29%)
    
    Returns:
        Meta score clamped to [0, 1]
    """
    # Convert winrate to percentage for calculation
    winrate_pct = winrate * 100.0
    base = (winrate_pct - 45.0) / 10.0 * META_WIN_WEIGHT
    base = max(0.0, base)  # Clamp negative values to 0
    ban_bonus = _calculate_ban_bonus(banrate)
    pick_penalty = _pickrate_penalty(pickrate)
    total = base + (ban_bonus * META_BAN_WEIGHT) - pick_penalty
    total = min(1.0, max(0.0, total))  # Clamp to [0, 1]
    return total


def build_meta_scores(winrates_path: Path, output_path: Path) -> None:
    """Build meta_scores.csv from u_gg_winrates.csv."""
    df = pd.read_csv(winrates_path)
    
    required = {"champion", "role", "winrate", "pickrate", "banrate"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    
    # All values in CSV are percentages, convert to decimals
    df["winrate_decimal"] = df["winrate"] / 100.0
    df["pickrate_decimal"] = df["pickrate"] / 100.0
    df["banrate_decimal"] = df["banrate"] / 100.0
    
    # Calculate meta score for each row
    meta_scores = []
    for _, row in df.iterrows():
        champion = str(row["champion"]).strip()
        role = str(row["role"]).strip().upper()
        winrate = float(row["winrate_decimal"])
        pickrate = float(row["pickrate_decimal"])
        banrate = float(row["banrate_decimal"])
        
        meta_score = calculate_meta_score(winrate, pickrate, banrate)
        
        meta_scores.append({
            "champion": champion,
            "role": role,
            "meta_score": meta_score,
        })
    
    # Create output DataFrame
    output_df = pd.DataFrame(meta_scores)
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV
    output_df.to_csv(output_path, index=False)
    print(f"Created {output_path} with {len(output_df)} entries")
    print(f"Sample entries:")
    print(output_df.head(10).to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description="Build meta_scores.csv from u_gg_winrates.csv")
    parser.add_argument(
        "--winrates",
        type=Path,
        default=Path("data/meta/u_gg_winrates.csv"),
        help="Path to u_gg_winrates.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/meta/soloq_meta_scores.csv"),
        help="Path to output soloq_meta_scores.csv",
    )
    args = parser.parse_args()
    
    build_meta_scores(args.winrates, args.output)


if __name__ == "__main__":
    main()
