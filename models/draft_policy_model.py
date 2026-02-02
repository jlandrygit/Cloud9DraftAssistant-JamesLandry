"""Baseline draft policy model for pick/ban decisions."""

from __future__ import annotations

from typing import Dict

import torch
from torch import nn


class DraftPolicyModel(nn.Module):
    """Baseline policy network producing logits per champion.

    Future work: player-comfort and opponent-threat features can be concatenated
    with the encoded draft-state vectors before the dense stack.
    """

    def __init__(
        self,
        *,
        champion_vocab_size: int,
        embed_dim: int = 64,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.champion_embed = nn.Embedding(champion_vocab_size, embed_dim)
        self.input_proj = nn.Linear(embed_dim * 4 + 2, hidden_dim)
        self.hidden = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.output = nn.Linear(hidden_dim, champion_vocab_size)

    def forward(
        self, features: Dict[str, torch.Tensor], action_mask: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass returning masked logits per champion."""
        # Aggregate champion vectors for available/picks/bans.
        avail = features["champion_available"]
        blue = features["blue_picks"]
        red = features["red_picks"]
        bans = features["bans"]

        # Convert one-hot vectors into weighted embeddings.
        weight = self.champion_embed.weight  # (vocab, embed_dim)
        avail_vec = avail @ weight
        blue_vec = blue @ weight
        red_vec = red @ weight
        bans_vec = bans @ weight

        pick_flag = features["is_pick_phase"]
        side_flag = features["acting_side"]

        x = torch.cat([avail_vec, blue_vec, red_vec, bans_vec, pick_flag, side_flag], dim=-1)
        x = self.input_proj(x)
        x = self.hidden(x)
        logits = self.output(x)

        # Mask illegal champions before softmax by assigning large negative logits.
        mask = action_mask.float()
        logits = logits.masked_fill(mask <= 0.0, -1e9)
        return logits
