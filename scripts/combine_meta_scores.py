"""Combine solo queue and pro play meta scores into final meta scores.

This script:
1. Loads soloq_meta_scores.csv (role-specific)
2. Loads pro_meta_scores.csv (role-agnostic)
3. Combines them: final_meta_score = 0.75 * pro_meta_score + 0.25 * soloq_meta_score
4. Saves to meta_scores.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def combine_meta_scores(
    soloq_path: Path,
    pro_path: Path,
    output_path: Path,
) -> None:
    """Combine solo queue and pro meta scores.
    
    Formula: final_meta_score = 0.75 * pro_meta_score + 0.25 * soloq_meta_score
    
    Since pro_meta_score is role-agnostic, use the same pro score for all roles.
    """
    # Load soloq scores (role-specific)
    if not soloq_path.exists():
        raise FileNotFoundError(f"Soloq meta scores not found: {soloq_path}")
    
    soloq_df = pd.read_csv(soloq_path)
    required = {"champion", "role", "meta_score"}
    missing = required - set(soloq_df.columns)
    if missing:
        raise ValueError(f"Soloq CSV missing columns: {sorted(missing)}")
    
    # Load pro scores (role-agnostic)
    pro_scores: dict[str, float] = {}
    if pro_path.exists():
        pro_df = pd.read_csv(pro_path)
        required_pro = {"champion", "pro_meta_score"}
        missing_pro = required_pro - set(pro_df.columns)
        if missing_pro:
            print(f"Warning: Pro CSV missing columns: {sorted(missing_pro)}")
        else:
            for _, row in pro_df.iterrows():
                champion = str(row["champion"]).strip()
                pro_score = float(row["pro_meta_score"])
                pro_scores[champion] = pro_score
    else:
        print(f"Warning: Pro meta scores not found: {pro_path}. Using 0.0 for all champions.")
    
    # Combine scores
    combined_scores = []
    for _, row in soloq_df.iterrows():
        champion = str(row["champion"]).strip()
        role = str(row["role"]).strip().upper()
        soloq_score = float(row["meta_score"])
        
        # Get pro score (default to 0.0 if not in pro data)
        pro_score = pro_scores.get(champion, 0.0)
        
        # Combine: 75% pro, 25% soloq
        final_score = (0.75 * pro_score) + (0.25 * soloq_score)
        
        combined_scores.append({
            "champion": champion,
            "role": role,
            "meta_score": final_score,
        })
    
    # Save to CSV
    output_df = pd.DataFrame(combined_scores)
    output_df = output_df.sort_values(["champion", "role"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)
    
    print(f"Combined {len(combined_scores)} champion/role combinations")
    print(f"Saved to {output_path}")
    print(f"\nSample scores:")
    sample = output_df.head(10)
    for _, row in sample.iterrows():
        print(f"  {row['champion']} {row['role']}: {row['meta_score']:.4f}")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Combine solo queue and pro meta scores"
    )
    parser.add_argument(
        "--soloq",
        type=str,
        default="data/meta/soloq_meta_scores.csv",
        help="Path to soloq_meta_scores.csv",
    )
    parser.add_argument(
        "--pro",
        type=str,
        default="data/meta/pro_meta_scores.csv",
        help="Path to pro_meta_scores.csv",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/meta/meta_scores.csv",
        help="Path to output meta_scores.csv",
    )
    args = parser.parse_args()
    
    combine_meta_scores(
        Path(args.soloq),
        Path(args.pro),
        Path(args.output),
    )


if __name__ == "__main__":
    main()
