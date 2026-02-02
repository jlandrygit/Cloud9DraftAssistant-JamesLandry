"""Quick verification script to check if setup is complete."""

from pathlib import Path
import sys

def check_file(path: str) -> tuple[bool, str]:
    """Check if a file exists."""
    p = Path(path)
    if p.exists():
        return True, "OK"
    return False, "MISSING"

def main():
    """Verify critical files and dependencies."""
    print("=" * 60)
    print("C9 Draft Assistant - Setup Verification")
    print("=" * 60)
    print(f"\nPython version: {sys.version.split()[0]}")
    print(f"Project root: {Path.cwd()}\n")
    
    print("Checking critical files...")
    critical_files = [
        "app.py",
        "api/main.py",
        "requirements.txt",
        "data/meta/meta_scores.csv",
        "data/meta/u_gg_roles.csv",
        "data/archetypes.json",
        "models/policy_checkpoints/draft_policy_epoch_5.pt",
    ]
    
    all_ok = True
    for file in critical_files:
        exists, status = check_file(file)
        print(f"  {file:50s} {status}")
        if not exists:
            all_ok = False
    
    print("\nChecking Python imports...")
    try:
        import streamlit
        print("  streamlit: OK")
    except ImportError:
        print("  streamlit: MISSING (run: pip install -r requirements.txt)")
        all_ok = False
    
    try:
        import fastapi
        print("  fastapi: OK")
    except ImportError:
        print("  fastapi: MISSING (run: pip install -r requirements.txt)")
        all_ok = False
    
    try:
        import torch
        print("  torch: OK")
    except ImportError:
        print("  torch: MISSING (run: pip install -r requirements.txt)")
        all_ok = False
    
    try:
        from core.config import DEFAULT_PATCH_VERSION
        print("  core.config: OK")
    except ImportError as e:
        print(f"  core.config: ERROR - {e}")
        all_ok = False
    
    try:
        from model.inference import recommend_picks, recommend_bans
        print("  model.inference: OK")
    except ImportError as e:
        print(f"  model.inference: ERROR - {e}")
        all_ok = False
    
    try:
        from data.ingestion import get_ingested_match_count
        print("  data.ingestion: OK")
    except ImportError as e:
        print(f"  data.ingestion: ERROR - {e}")
        all_ok = False
    
    print("\n" + "=" * 60)
    if all_ok:
        print("[OK] Setup looks good! You can run the demo.")
        print("\nTo start:")
        print("  1. uvicorn api.main:app --reload")
        print("  2. streamlit run app.py")
    else:
        print("[ERROR] Some issues found. Please fix them before running the demo.")
    print("=" * 60)

if __name__ == "__main__":
    main()
