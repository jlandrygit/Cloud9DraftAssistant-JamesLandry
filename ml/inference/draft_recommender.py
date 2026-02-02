"""Draft policy inference helpers."""

from __future__ import annotations

import ast
import hashlib
import os
from pathlib import Path
from functools import lru_cache
from typing import Dict

import joblib
import numpy as np
import pandas as pd

try:
    import torch
except Exception:  # pragma: no cover - torch may be unavailable in demo envs
    torch = None

from core.scoring_draft_state import DraftState
from ml.features.draft_encoder import DraftStateEncoder
from models.draft_policy_model import DraftPolicyModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=1)
def _load_ban_rate_bonus() -> Dict[str, float]:
    """Load ban rate data to reward highly-banned champions.
    
    Champions with high ban rates are usually very strong, so if they're available
    to pick, they should be rewarded even if they have low pick counts (because
    they're usually banned).
    
    Returns:
        Dict mapping normalized champion name to ban rate bonus multiplier (1.0-2.0).
        - 0% ban rate: 1.0 (no bonus)
        - 50%+ ban rate: up to 1.5 (50% bonus)
        - 80%+ ban rate: up to 2.0 (100% bonus)
    """
    try:
        pro_meta_path = PROJECT_ROOT / "data" / "meta" / "gol_gg_pro_meta.csv"
        if not pro_meta_path.exists():
            return {}
        
        df = pd.read_csv(pro_meta_path)
        TOTAL_GAMES = 201  # Total pro games this patch
        ban_bonus_map: Dict[str, float] = {}
        
        for _, row in df.iterrows():
            champion = str(row["champion"]).strip()
            bans = int(row.get("bans", 0))
            picks = int(row.get("picks", 0))
            
            # Normalize champion name
            normalized_champion = champion.replace(" ", "").replace("'", "").replace(".", "").replace("&", "")
            
            # Calculate ban rate percentage
            banrate_pct = (bans / TOTAL_GAMES) * 100.0 if TOTAL_GAMES > 0 else 0.0
            
            # Calculate bonus: scales from 1.0 (0% ban rate) to 2.0 (100% ban rate)
            # More aggressive scaling for high ban rates
            if banrate_pct >= 80.0:
                # 80%+ ban rate: 1.5 to 2.0 bonus (50% to 100% boost)
                bonus = 1.5 + ((banrate_pct - 80.0) / 20.0) * 0.5
            elif banrate_pct >= 50.0:
                # 50-80% ban rate: 1.2 to 1.5 bonus (20% to 50% boost)
                bonus = 1.2 + ((banrate_pct - 50.0) / 30.0) * 0.3
            elif banrate_pct >= 25.0:
                # 25-50% ban rate: 1.05 to 1.2 bonus (5% to 20% boost)
                bonus = 1.05 + ((banrate_pct - 25.0) / 25.0) * 0.15
            elif banrate_pct >= 10.0:
                # 10-25% ban rate: 1.0 to 1.05 bonus (0% to 5% boost)
                bonus = 1.0 + ((banrate_pct - 10.0) / 15.0) * 0.05
            else:
                # <10% ban rate: no bonus
                bonus = 1.0
            
            ban_bonus_map[normalized_champion] = min(2.0, bonus)
        
        return ban_bonus_map
    except Exception:
        return {}


@lru_cache(maxsize=1)
def _load_pro_play_frequency() -> Dict[str, float]:
    """Load pro play frequency data to penalize champions with low presence.
    
    Returns:
        Dict mapping champion name to presence-based frequency multiplier (0.0-1.0).
        Penalty is based on total presence (picks + bans) from gol_gg_pro_meta.csv:
        - 0 presence: 0.1 multiplier (90% penalty)
        - 1 presence: 0.25 multiplier (75% penalty) - starting point
        - Scales up from 0.25 to 1.0 as total presence increases
    """
    try:
        # Load from gol_gg_pro_meta.csv
        pro_meta_path = PROJECT_ROOT / "data" / "meta" / "gol_gg_pro_meta.csv"
        if pro_meta_path.exists():
            df = pd.read_csv(pro_meta_path)
            frequency_map: Dict[str, float] = {}
            for _, row in df.iterrows():
                champion = str(row["champion"]).strip()
                picks = int(row.get("picks", 0))
                bans = int(row.get("bans", 0))
                # Total presence = picks + bans (how often the champion appears in drafts)
                total_presence = picks + bans
                
                # Normalize champion name to match the format used in get_policy_scores
                # (remove spaces, apostrophes, periods, &)
                normalized_champion = champion.replace(" ", "").replace("'", "").replace(".", "").replace("&", "")
                
                # Penalty scaling based on total presence (picks + bans)
                if total_presence == 0:
                    frequency_map[normalized_champion] = 0.1  # 90% penalty for never picked/banned
                elif total_presence == 1:
                    frequency_map[normalized_champion] = 0.25  # 75% penalty for 1 presence (starting point)
                else:
                    # Scale from 0.25 (1 presence) to 1.0 (50+ presence)
                    # Linear interpolation: 1 presence -> 0.25, 50 presence -> 1.0
                    frequency_map[normalized_champion] = min(1.0, 0.25 + (min(total_presence - 1, 49) / 49.0 * 0.75))
            
            return frequency_map
    except Exception:
        pass
    
    # Default: no frequency data available, return empty dict (no penalty)
    return {}


@lru_cache(maxsize=1)
def _load_champion_vocab() -> list[str]:
    """Load the champion vocabulary used for policy scoring."""
    champions: set[str] = set()
    try:
        encoder = joblib.load("data/processed/champion_encoder.pkl")
        champions.update(list(encoder.classes_))
    except Exception:
        pass
    champions.update(_load_champions_from_app())
    return sorted(champions)


def _load_champions_from_app() -> list[str]:
    """Parse the CHAMPIONS list from app.py without importing Streamlit."""
    app_path = PROJECT_ROOT / "app.py"
    if not app_path.exists():
        return []
    text = app_path.read_text(encoding="utf-8")
    marker = "CHAMPIONS = ["
    start = text.find(marker)
    if start == -1:
        return []
    start = text.find("[", start)
    if start == -1:
        return []
    bracket_depth = 0
    end = None
    for idx in range(start, len(text)):
        char = text[idx]
        if char == "[":
            bracket_depth += 1
        elif char == "]":
            bracket_depth -= 1
            if bracket_depth == 0:
                end = idx + 1
                break
    if end is None:
        return []
    snippet = text[start:end]
    try:
        parsed = ast.literal_eval(snippet)
    except Exception:
        return []
    if not isinstance(parsed, list):
        return []
    return [str(item) for item in parsed if isinstance(item, str)]


def get_policy_scores(state: DraftState, include_tier2: bool = False) -> Dict[str, float]:
    """Return per-champion policy scores in [0, 1] without global normalization.

    Scores are deterministic and scaled independently via a sigmoid so they
    can be compared directly against other signals in the final score.
    Masked champions (already picked/banned) are excluded from the output.
    
    Args:
        state: Current draft state
        include_tier2: If True, use model trained on Tier 1 + Tier 2 data, else use Tier 1 only
    """
    champions = _load_champion_vocab()
    if not champions:
        return {}

    available = state.available_champions(set(champions))
    if not available:
        return {}

    ordered = sorted(available)
    model_logits = _get_model_logits(state, include_tier2=include_tier2)
    logits = np.array(
        [
            model_logits.get(champ, _stable_logit(champ))
            for champ in ordered
        ],
        dtype=np.float64,
    )
    scores = _sigmoid(logits)
    
    # Apply frequency-based penalty for rarely-picked champions
    frequency_map = _load_pro_play_frequency()
    ban_bonus_map = _load_ban_rate_bonus()
    
    penalized_scores: Dict[str, float] = {}
    for champ, score in zip(ordered, scores):
        # Normalize champion name to match CSV format (remove spaces, apostrophes, periods)
        normalized_champ = champ.replace(" ", "").replace("'", "").replace(".", "").replace("&", "")
        
        # Apply frequency penalty
        frequency_mult = frequency_map.get(normalized_champ, 1.0)  # Default: no penalty if not in map
        penalized_score = score * frequency_mult
        
        # Apply ban rate bonus (especially important for high-ban, low-pick champions)
        # This rewards champions that are usually banned (like Jayce, Malphite)
        ban_bonus = ban_bonus_map.get(normalized_champ, 1.0)  # Default: no bonus if not in map
        penalized_score = penalized_score * ban_bonus
        
        # Clamp final score to [0, 1] to ensure it stays within bounds
        penalized_score = max(0.0, min(1.0, penalized_score))
        
        penalized_scores[champ] = float(penalized_score)
    
    return penalized_scores


def _get_model_logits(state: DraftState, include_tier2: bool = False) -> Dict[str, float]:
    """Return model logits keyed by champion, if a policy model is available.
    
    Args:
        state: Current draft state
        include_tier2: If True, use model trained on Tier 1 + Tier 2 data, else use Tier 1 only
    """
    if torch is None:
        return {}
    checkpoint = _load_policy_checkpoint(include_tier2=include_tier2)
    if not checkpoint:
        return {}
    model, vocab = checkpoint
    encoder = DraftStateEncoder.from_vocab(
        champions=vocab.get("champions", []),
        patches=vocab.get("patches", [state.patch]),
        leagues=vocab.get("leagues", [state.league]),
    )
    features = encoder.encode(state)
    action_mask = features["champion_available"]
    with torch.no_grad():
        feature_tensors = {
            key: torch.from_numpy(value.astype(np.float32)).unsqueeze(0)
            for key, value in features.items()
            if key != "step_index"
        }
        mask_tensor = torch.from_numpy(action_mask.astype(np.float32)).unsqueeze(0)
        logits = model(feature_tensors, mask_tensor).squeeze(0).cpu().numpy()
    champions = vocab.get("champions", [])
    return {champ: float(logits[idx]) for idx, champ in enumerate(champions)}


def _load_policy_checkpoint(include_tier2: bool = False) -> tuple[DraftPolicyModel, dict] | None:
    """Load the policy model checkpoint if available.
    
    Args:
        include_tier2: If True, load Tier 2 model (draft_policy_t2), else load Tier 1 model (draft_policy)
    """
    checkpoint_path = _resolve_checkpoint_path(include_tier2=include_tier2)
    if not checkpoint_path:
        return None
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    except Exception:
        return None
    vocab = checkpoint.get("encoder_vocab", {})
    champions = vocab.get("champions", [])
    if not champions:
        return None
    model = DraftPolicyModel(champion_vocab_size=len(champions))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, vocab


def _resolve_checkpoint_path(include_tier2: bool = False) -> Path | None:
    """Find the latest policy checkpoint or use POLICY_MODEL_PATH.
    
    Args:
        include_tier2: If True, look for Tier 2 model (draft_policy_t2), else Tier 1 model (draft_policy)
    """
    env_value = os.getenv("POLICY_MODEL_PATH")
    env_path = PROJECT_ROOT / Path(env_value) if env_value else None
    if env_path and env_path.exists():
        return env_path
    checkpoint_dir = PROJECT_ROOT / "models" / "policy_checkpoints"
    if not checkpoint_dir.exists():
        return None
    # Select model pattern based on tier2 flag
    pattern = "draft_policy_t2_epoch_*.pt" if include_tier2 else "draft_policy_epoch_*.pt"
    candidates = sorted(checkpoint_dir.glob(pattern))
    return candidates[-1] if candidates else None


def _stable_logit(champion: str) -> float:
    """Deterministic pseudo-logit for demo policy scoring."""
    digest = hashlib.sha1(champion.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid for independent scoring."""
    logits = np.clip(logits, -20, 20)
    return 1.0 / (1.0 + np.exp(-logits))
