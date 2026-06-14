"""Run 5-fold search over EfficientNet fine-tuning depth and hidden size."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from sklearn.model_selection import KFold

from mai_spatial.data import read_expression_csv
from mai_spatial.engine import seed_everything
from mai_spatial.experiments import TrainConfig, fit_single_split, summarize_score_row


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-csv", default="dataset.csv")
    parser.add_argument("--out-dir", default="runs/finetuning_search")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--hidden-grid", default="1024,3000,4096")
    parser.add_argument("--unfreeze-grid", default="-9,0")
    parser.add_argument("--augmentation", default="mpa", choices=["mpa", "stnet", "none"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def parse_int_grid(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df, gene_columns = read_expression_csv(args.data_csv)
    kfold = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)

    fold_rows = []
    summary_rows = []

    for unfreeze_from in parse_int_grid(args.unfreeze_grid):
        for hidden_dim in parse_int_grid(args.hidden_grid):
            fold_scores = []
            fold_losses = []
            print(f"\n=== hidden_dim={hidden_dim} unfreeze_from={unfreeze_from} ===")

            for fold, (train_index, valid_index) in enumerate(kfold.split(df), start=1):
                config = TrainConfig(
                    epochs=args.epochs,
                    learning_rate=args.learning_rate,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    seed=args.seed + fold,
                    augmentation=args.augmentation,
                    hidden_dim=hidden_dim,
                    unfreeze_from=unfreeze_from,
                )
                _, valid_loss, scores = fit_single_split(
                    train_df=df.iloc[train_index],
                    valid_df=df.iloc[valid_index],
                    gene_columns=gene_columns,
                    backbone_name="efficientnet_b0",
                    config=config,
                    device=device,
                    checkpoint_path=None,
                )
                fold_rows.append(
                    summarize_score_row(
                        {
                            "hidden_dim": hidden_dim,
                            "unfreeze_from": unfreeze_from,
                            "fold": fold,
                        },
                        valid_loss,
                        scores,
                    )
                )
                fold_scores.append(scores)
                fold_losses.append(valid_loss)

            summary_rows.append(
                {
                    "hidden_dim": hidden_dim,
                    "unfreeze_from": unfreeze_from,
                    "valid_loss": sum(fold_losses) / len(fold_losses),
                    "HEG": sum(score["HEG"] for score in fold_scores) / len(fold_scores),
                    "HVG": sum(score["HVG"] for score in fold_scores) / len(fold_scores),
                    "MG": sum(score["MG"] for score in fold_scores) / len(fold_scores),
                }
            )

    fold_results = pd.DataFrame(fold_rows)
    summary = pd.DataFrame(summary_rows).sort_values(["MG", "HVG", "HEG"], ascending=False)
    fold_results.to_csv(out_dir / "finetuning_fold_results.csv", index=False)
    summary.to_csv(out_dir / "finetuning_grid_summary.csv", index=False)
    print(summary)
    print(f"Saved: {out_dir / 'finetuning_grid_summary.csv'}")


if __name__ == "__main__":
    main()
