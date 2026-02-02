"""Train a baseline draft policy model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from core.scoring_draft_state import DraftState
from ml.features.draft_encoder import DraftStateEncoder
from models.draft_policy_model import DraftPolicyModel


class DraftDatasetWithPhase(Dataset):
    """Dataset returning encoded features, target, mask, and action type."""

    def __init__(self, df: pd.DataFrame, encoder: DraftStateEncoder) -> None:
        self._df = df.reset_index(drop=True)
        self._encoder = encoder

    def __len__(self) -> int:
        return len(self._df)

    def __getitem__(self, idx: int) -> tuple[dict[str, torch.Tensor], int, torch.Tensor, str]:
        row = self._df.iloc[idx]
        state = _state_from_row(row)
        encoded = self._encoder.encode(state)
        features = {k: torch.from_numpy(v.astype(np.float32)) for k, v in encoded.items()}

        champion_name = str(row["action_champion"])
        target = int(self._encoder.champion_encoder.transform([champion_name])[0])
        mask = torch.from_numpy(encoded["champion_available"].astype(np.float32))
        action_type = str(row["action_type"]).upper()
        return features, target, mask, action_type


def main() -> None:
    parser = argparse.ArgumentParser(description="Train draft policy model.")
    parser.add_argument(
        "--input",
        default="data/processed/exploded_drafts.parquet",
        help="Path to exploded draft decisions",
    )
    parser.add_argument(
        "--output",
        default="models/policy_checkpoints",
        help="Directory for model checkpoints",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--model-name",
        default="draft_policy",
        help="Base name for model checkpoints (e.g., 'draft_policy' or 'draft_policy_t2')",
    )
    args = parser.parse_args()

    df = _load_dataframe(args.input)
    df = df[df["action_champion"].notna() & (df["action_champion"] != "")]
    df = df[~df["action_champion"].str.strip().str.upper().isin({"B", "R"})]

    patches = sorted(df["draft_state"].map(lambda s: s.get("patch", "")).unique())
    latest_patch = patches[-1] if patches else ""
    recent_df = df[df["draft_state"].map(lambda s: s.get("patch", "")) == latest_patch]
    if recent_df.empty:
        recent_df = df
    train_df, val_df = train_test_split(
        recent_df, test_size=0.2, random_state=42, shuffle=True
    )

    champions = sorted(set(df["action_champion"].astype(str)))
    patches_vocab = sorted(_collect_vocab(df, key="patch"))
    leagues_vocab = sorted(_collect_vocab(df, key="league"))
    encoder = DraftStateEncoder.from_vocab(champions, patches_vocab, leagues_vocab)

    train_ds = DraftDatasetWithPhase(train_df, encoder)
    val_ds = DraftDatasetWithPhase(val_df, encoder)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = DraftPolicyModel(champion_vocab_size=len(encoder.champion_encoder.classes_))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_metrics = _run_epoch(model, train_loader, optimizer=optimizer)
        val_metrics = _run_epoch(model, val_loader, optimizer=None)
        _log_metrics(epoch, train_metrics, val_metrics)

        checkpoint = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "encoder_vocab": {
                "champions": champions,
                "patches": patches_vocab,
                "leagues": leagues_vocab,
            },
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
        }
        torch.save(checkpoint, output_dir / f"{args.model_name}_epoch_{epoch}.pt")


def _run_epoch(
    model: DraftPolicyModel,
    loader: DataLoader,
    *,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, float]:
    model.train(optimizer is not None)
    total_loss = 0.0
    total = 0
    top3 = 0
    top5 = 0
    pick_correct = 0
    pick_total = 0
    ban_correct = 0
    ban_total = 0

    loss_fn = nn.CrossEntropyLoss()

    for features, target, mask, action_type in loader:
        logits = model(features, mask)
        loss = loss_fn(logits, target)

        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * target.size(0)
        total += target.size(0)

        _, top3_idx = logits.topk(3, dim=1)
        _, top5_idx = logits.topk(5, dim=1)
        target_exp = target.unsqueeze(1)
        top3 += (top3_idx == target_exp).any(dim=1).sum().item()
        top5 += (top5_idx == target_exp).any(dim=1).sum().item()

        preds = logits.argmax(dim=1)
        for pred, label, phase in zip(preds, target, action_type):
            if phase == "PICK":
                pick_total += 1
                if pred.item() == label.item():
                    pick_correct += 1
            elif phase == "BAN":
                ban_total += 1
                if pred.item() == label.item():
                    ban_correct += 1

    return {
        "loss": total_loss / max(1, total),
        "top3_acc": top3 / max(1, total),
        "top5_acc": top5 / max(1, total),
        "pick_acc": pick_correct / max(1, pick_total),
        "ban_acc": ban_correct / max(1, ban_total),
    }


def _log_metrics(epoch: int, train: dict[str, float], val: dict[str, float]) -> None:
    print(
        f"Epoch {epoch} | "
        f"train_loss={train['loss']:.4f} "
        f"val_loss={val['loss']:.4f} "
        f"train_top3={train['top3_acc']:.3f} "
        f"val_top3={val['top3_acc']:.3f} "
        f"train_top5={train['top5_acc']:.3f} "
        f"val_top5={val['top5_acc']:.3f} "
        f"train_pick={train['pick_acc']:.3f} "
        f"val_pick={val['pick_acc']:.3f} "
        f"train_ban={train['ban_acc']:.3f} "
        f"val_ban={val['ban_acc']:.3f}"
    )


def _state_from_row(row: pd.Series) -> DraftState:
    payload = row["draft_state"]
    return DraftState(
        patch=str(payload["patch"]),
        league=str(payload["league"]),
        blue_team=str(payload["blue_team"]),
        red_team=str(payload["red_team"]),
        blue_picks=set(payload["blue_picks"]),
        red_picks=set(payload["red_picks"]),
        bans=set(payload["bans"]),
        step_index=int(payload["step_index"]),
    )


def _collect_vocab(df: pd.DataFrame, *, key: str) -> set[str]:
    values = set()
    for payload in df["draft_state"]:
        values.add(str(payload.get(key, "")))
    return values


def _load_dataframe(path: str) -> pd.DataFrame:
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path, converters={"draft_state": json.loads})


if __name__ == "__main__":
    main()
