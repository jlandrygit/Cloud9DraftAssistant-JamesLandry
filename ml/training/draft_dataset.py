"""PyTorch dataset for draft decision modeling."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from core.scoring_draft_state import DraftState
from ml.features.draft_encoder import DraftStateEncoder


class DraftDecisionDataset(Dataset):
    """Dataset providing encoded draft states and action labels."""

    def __init__(
        self, df: pd.DataFrame, encoder: DraftStateEncoder
    ) -> None:
        self._df = df.reset_index(drop=True)
        self._encoder = encoder

    def __len__(self) -> int:
        return len(self._df)

    def __getitem__(self, idx: int) -> tuple[dict[str, torch.Tensor], int, torch.Tensor]:
        row = self._df.iloc[idx]
        state = _state_from_row(row)
        encoded = self._encoder.encode(state)

        feature_dict = {key: torch.from_numpy(value.astype(np.float32)) for key, value in encoded.items()}

        champion_name = str(row["action_champion"])
        if champion_name not in self._encoder.champion_encoder.classes_:
            raise ValueError(f"Unknown champion in label: {champion_name}")
        target = int(self._encoder.champion_encoder.transform([champion_name])[0])

        available = encoded["champion_available"].astype(np.float32)
        action_mask = torch.from_numpy(available)
        return feature_dict, target, action_mask


def _state_from_row(row: pd.Series) -> DraftState:
    """Reconstruct DraftState from a serialized row."""
    state_payload = row["draft_state"]
    return DraftState(
        patch=str(state_payload["patch"]),
        league=str(state_payload["league"]),
        blue_team=str(state_payload["blue_team"]),
        red_team=str(state_payload["red_team"]),
        blue_picks=set(state_payload["blue_picks"]),
        red_picks=set(state_payload["red_picks"]),
        bans=set(state_payload["bans"]),
        step_index=int(state_payload["step_index"]),
    )
