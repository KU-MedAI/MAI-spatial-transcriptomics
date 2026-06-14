"""Train the EfficientNet-B0 single-patch baseline."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn

from mai_spatial.data import GeneExpressionDataset, read_expression_csv, sort_spots, split_train_valid
from mai_spatial.engine import evaluate, predict, seed_everything
from mai_spatial.experiments import TrainConfig, fit_single_split, make_dataloader
from mai_spatial.transforms import build_transforms


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-csv", default="dataset.csv")
    parser.add_argument("--test-csv", default=None)
    parser.add_argument("--out-dir", default="runs/effb0_baseline")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden-dim", type=int, default=3000)
    parser.add_argument("--unfreeze-from", type=int, default=-9)
    parser.add_argument("--augmentation", default="mpa", choices=["mpa", "stnet", "none"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df, gene_columns = read_expression_csv(args.data_csv)
    train_df, valid_df = split_train_valid(df, seed=args.seed)
    config = TrainConfig(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        augmentation=args.augmentation,
        hidden_dim=args.hidden_dim,
        unfreeze_from=args.unfreeze_from,
    )

    model, best_loss, best_scores = fit_single_split(
        train_df=train_df,
        valid_df=valid_df,
        gene_columns=gene_columns,
        backbone_name="efficientnet_b0",
        config=config,
        device=device,
        checkpoint_path=out_dir / "best_effb0.pth",
    )
    print(f"[best] valid_loss={best_loss:.5f} scores={best_scores}")

    if args.test_csv is None:
        return

    test_df = sort_spots(pd.read_csv(args.test_csv))
    _, eval_transform = build_transforms(args.augmentation)
    test_has_labels = all(column in test_df.columns for column in gene_columns)
    test_dataset = GeneExpressionDataset(
        test_df,
        gene_columns,
        eval_transform,
        has_labels=test_has_labels,
    )
    test_loader = make_dataloader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        shuffle=False,
    )

    if test_has_labels:
        loss, scores = evaluate(model, test_loader, nn.MSELoss().to(device), device, gene_columns)
        print(f"[test] loss={loss:.5f} scores={scores}")
    else:
        predictions = predict(model, test_loader, device)
        output = pd.DataFrame(predictions, columns=gene_columns)
        if "id" in test_df.columns:
            output.insert(0, "id", test_df["id"].values)
        output.to_csv(out_dir / "pred_test.csv", index=False)
        print(f"Saved predictions: {out_dir / 'pred_test.csv'}")


if __name__ == "__main__":
    main()
