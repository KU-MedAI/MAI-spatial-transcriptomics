"""Training and inference loops."""

from __future__ import annotations

import os
import random
from contextlib import nullcontext

import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from mai_spatial.metrics import grouped_pearson_corr


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def seed_worker(worker_id: int, base_seed: int = 42) -> None:
    worker_seed = base_seed + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def autocast_for(device: torch.device):
    if device.type == "cuda":
        return autocast()
    return nullcontext()


def train_one_epoch(model, loader, optimizer, scaler, criterion, device):
    model.train()
    losses = []

    for images, labels in tqdm(loader, desc="Train", leave=False):
        images = images.to(device).float()
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        with autocast_for(device):
            predictions = model(images)
            loss = criterion(predictions, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        losses.append(loss.item())

    return float(np.mean(losses))


@torch.no_grad()
def evaluate(model, loader, criterion, device, gene_columns):
    model.eval()
    losses = []
    predictions = []
    targets = []

    for images, labels in tqdm(loader, desc="Valid", leave=False):
        images = images.to(device).float()
        labels = labels.to(device)

        with autocast_for(device):
            batch_predictions = model(images)
            loss = criterion(batch_predictions, labels)

        losses.append(loss.item())
        predictions.append(batch_predictions.float().cpu().numpy())
        targets.append(labels.float().cpu().numpy())

    pred = np.concatenate(predictions, axis=0)
    true = np.concatenate(targets, axis=0)
    return float(np.mean(losses)), grouped_pearson_corr(pred, true, gene_columns)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    predictions = []

    for batch in tqdm(loader, desc="Infer", leave=False):
        images = batch[0] if isinstance(batch, (list, tuple)) else batch
        images = images.to(device).float()

        with autocast_for(device):
            batch_predictions = model(images)

        predictions.append(batch_predictions.float().cpu().numpy())

    return np.concatenate(predictions, axis=0)


def make_grad_scaler(device: torch.device) -> GradScaler:
    return GradScaler(enabled=device.type == "cuda")
