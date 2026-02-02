"""DraftState encoder for model-agnostic feature generation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.preprocessing import LabelEncoder

from core.draft_order import DRAFT_SEQUENCE
from core.scoring_draft_state import DraftState


@dataclass
class DraftStateEncoder:
    """Encode DraftState into deterministic numeric feature vectors."""

    champion_encoder: LabelEncoder
    patch_encoder: LabelEncoder
    league_encoder: LabelEncoder

    @classmethod
    def from_vocab(
        cls, champions: list[str], patches: list[str], leagues: list[str]
    ) -> "DraftStateEncoder":
        """Build encoders from ordered vocabularies."""
        champion_encoder = LabelEncoder()
        patch_encoder = LabelEncoder()
        league_encoder = LabelEncoder()

        champion_encoder.fit(sorted(champions))
        patch_encoder.fit(sorted(patches))
        league_encoder.fit(sorted(leagues))
        return cls(champion_encoder, patch_encoder, league_encoder)

    def encode(self, state: DraftState) -> dict[str, np.ndarray]:
        """Encode DraftState into a dictionary of numpy arrays."""
        champions = list(self.champion_encoder.classes_)
        size = len(champions)

        available = np.zeros(size, dtype=np.float32)
        blue = np.zeros(size, dtype=np.float32)
        red = np.zeros(size, dtype=np.float32)
        bans = np.zeros(size, dtype=np.float32)

        available_set = state.available_champions(set(champions))
        for champ in available_set:
            if champ in self.champion_encoder.classes_:
                idx = self.champion_encoder.transform([champ])[0]
                available[idx] = 1.0

        for champ in state.blue_picks:
            if champ in self.champion_encoder.classes_:
                idx = self.champion_encoder.transform([champ])[0]
                blue[idx] = 1.0

        for champ in state.red_picks:
            if champ in self.champion_encoder.classes_:
                idx = self.champion_encoder.transform([champ])[0]
                red[idx] = 1.0

        for champ in state.bans:
            if champ in self.champion_encoder.classes_:
                idx = self.champion_encoder.transform([champ])[0]
                bans[idx] = 1.0

        acting = 1.0 if state.acting_side() == "BLUE" else 0.0
        step_norm = state.step_index / max(1, len(DRAFT_SEQUENCE) - 1)

        return {
            "champion_available": available,
            "blue_picks": blue,
            "red_picks": red,
            "bans": bans,
            "is_pick_phase": np.array([1.0 if state.is_pick_phase() else 0.0]),
            "acting_side": np.array([acting]),
            "step_index": np.array([step_norm], dtype=np.float32),
        }
