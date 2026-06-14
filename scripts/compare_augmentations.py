"""Compare augmentation policies on the EfficientNet-B0 baseline."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from mai_spatial.data import read_expression_csv, split_train_valid
from mai_spatial.engine import seed_everything
from mai_spatial.experiments import TrainConfig, fit_single_split, summarize_score_row


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-csv", default="dataset.csv")
    parser.add_argument("--out-dir", default="runs/augmentation_comparison")
    parser.add_argument("--augmentations", default="mpa,stnet,none")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden-dim", type=int, default=3000)
    parser.add_argument("--unfreeze-from", type=int, default=-9)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df, gene_columns = read_expression_csv(args.data_csv)
    train_df, valid_df = split_train_valid(df, seed=args.seed)

    rows = []
    for augmentation in [item.strip() for item in args.augmentations.split(",") if item.strip()]:
        print(f"\n=== Augmentation: {augmentation} ===")
        config = TrainConfig(
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            seed=args.seed,
            augmentation=augmentation,
            hidden_dim=args.hidden_dim,
            unfreeze_from=args.unfreeze_from,
        )
        _, valid_loss, scores = fit_single_split(
            train_df=train_df,
            valid_df=valid_df,
            gene_columns=gene_columns,
            backbone_name="efficientnet_b0",
            config=config,
            device=device,
            checkpoint_path=out_dir / f"best_effb0_{augmentation}.pth",
        )
        rows.append(summarize_score_row({"augmentation": augmentation}, valid_loss, scores))

    results = pd.DataFrame(rows).sort_values(["MG", "HVG", "HEG"], ascending=False)
    results.to_csv(out_dir / "augmentation_comparison.csv", index=False)
    print(results)
    print(f"Saved: {out_dir / 'augmentation_comparison.csv'}")


if __name__ == "__main__":
    main()
