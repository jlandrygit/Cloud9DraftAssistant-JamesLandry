"""Script to explode gol_pro_draft_data.xlsx into exploded_drafts.parquet."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.explode_drafts import explode_all_drafts
from data.load_drafts_excel import load_draft_excel
from app import CHAMPIONS


def main() -> None:
    """Load Excel file, explode drafts, and save to parquet."""
    input_path = ROOT / "gol_pro_draft_data.xlsx"
    output_path = ROOT / "data" / "processed" / "t2_inc_exploded_drafts.parquet"
    
    print(f"Loading drafts from {input_path}...")
    df = load_draft_excel(str(input_path))
    print(f"Loaded {len(df)} draft rows")
    
    print("Exploding drafts into step-level decisions...")
    all_champions = set(CHAMPIONS)
    exploded_df = explode_all_drafts(df, all_champions)
    print(f"Exploded into {len(exploded_df)} step-level decisions")
    
    print(f"Saving to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    exploded_df.to_parquet(output_path, index=False)
    print(f"Saved {len(exploded_df)} records to {output_path}")


if __name__ == "__main__":
    main()
