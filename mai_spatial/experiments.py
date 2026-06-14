"""Reusable experiment runners."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from mai_spatial.data import GeneExpressionDataset
from mai_spatial.engine import evaluate, make_grad_scaler, seed_worker, train_one_epoch
from mai_spatial.models import SinglePatchRegressor
from mai_spatial.transforms import build_transforms


@dataclass
class TrainConfig:
    image_size: int = 224
    epochs: int = 30
    learning_rate: float = 3e-4
    batch_size: int = 32
    num_workers: int = 4
    seed: int = 42
    augmentation: str = "mpa"
    hidden_dim: int = 3000
    unfreeze_from: int | None = -9
    pretrained: bool = True


def make_dataloader(
    dataset,
    batch_size: int,
    num_workers: int,
    seed: int,
    shuffle: bool,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        worker_init_fn=partial(seed_worker, base_seed=seed),
        generator=generator,
    )


def fit_single_split(
    train_df,
    valid_df,
    gene_columns: list[str],
    backbone_name: str,
    config: TrainConfig,
    device: torch.device,
    checkpoint_path: str | Path | None = None,
):
    train_transform, eval_transform = build_transforms(config.augmentation, config.image_size)
    train_dataset = GeneExpressionDataset(train_df, gene_columns, train_transform, has_labels=True)
    valid_dataset = GeneExpressionDataset(valid_df, gene_columns, eval_transform, has_labels=True)

    train_loader = make_dataloader(
        train_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        seed=config.seed,
        shuffle=True,
    )
    valid_loader = make_dataloader(
        valid_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        seed=config.seed,
        shuffle=False,
    )

    model = SinglePatchRegressor(
        out_dim=len(gene_columns),
        backbone_name=backbone_name,
        hidden_dim=config.hidden_dim,
        unfreeze_from=config.unfreeze_from,
        pretrained=config.pretrained,
    ).to(device)
    optimizer = optim.Adam(
        filter(lambda parameter: parameter.requires_grad, model.parameters()),
        lr=config.learning_rate,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    criterion = nn.MSELoss().to(device)
    scaler = make_grad_scaler(device)

    best_loss = float("inf")
    best_scores: dict[str, float] = {}
    best_state = None

    for epoch in range(1, config.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, criterion, device)
        valid_loss, scores = evaluate(model, valid_loader, criterion, device, gene_columns)
        scheduler.step()

        print(
            f"[{epoch:02d}] train={train_loss:.5f} valid={valid_loss:.5f} "
            f"HEG={scores['HEG']:.4f} HVG={scores['HVG']:.4f} MG={scores['MG']:.4f}"
        )

        if valid_loss < best_loss:
            best_loss = valid_loss
            best_scores = scores
            best_state = {key: value.cpu() for key, value in model.state_dict().items()}
            if checkpoint_path is not None:
                checkpoint_path = Path(checkpoint_path)
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(best_state, checkpoint_path)

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, float(best_loss), best_scores


def summarize_score_row(prefix: dict, valid_loss: float, scores: dict[str, float]) -> dict:
    return {
        **prefix,
        "valid_loss": float(valid_loss),
        "HEG": float(scores.get("HEG", np.nan)),
        "HVG": float(scores.get("HVG", np.nan)),
        "MG": float(scores.get("MG", np.nan)),
    }
