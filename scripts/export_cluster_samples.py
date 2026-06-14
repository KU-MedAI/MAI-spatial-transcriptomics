"""Export cluster sample galleries from a cluster assignment CSV."""

from __future__ import annotations

import argparse
import math
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from mai_spatial.clustering import plot_cluster_embedding, plot_tissue_distribution


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cluster-csv",
        default="runs/cluster_weight_search/cluster_assignments.csv",
    )
    parser.add_argument("--out-dir", default="runs/cluster_samples")
    parser.add_argument("--samples-per-cluster", type=int, default=50)
    parser.add_argument("--gallery-size", type=int, default=25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = pd.read_csv(args.cluster_csv)
    out_dir = Path(args.out_dir)
    sample_dir = out_dir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)

    plot_cluster_embedding(results, out_dir / "cluster_embedding.png")
    plot_tissue_distribution(results, out_dir / "tissue_distribution.png")
    export_samples(results, sample_dir, args.samples_per_cluster)
    export_galleries(results, out_dir / "galleries", args.gallery_size)
    print(f"Saved cluster report: {out_dir}")


def export_samples(results: pd.DataFrame, out_dir: Path, samples_per_cluster: int) -> None:
    for cluster_id, cluster_df in results.groupby("cluster"):
        cluster_dir = out_dir / f"cluster_{cluster_id}"
        cluster_dir.mkdir(parents=True, exist_ok=True)
        for _, row in cluster_df.sample(
            n=min(samples_per_cluster, len(cluster_df)),
            random_state=42,
        ).iterrows():
            src = Path(row["image_path"])
            if src.exists():
                shutil.copy2(src, cluster_dir / src.name)


def export_galleries(results: pd.DataFrame, out_dir: Path, gallery_size: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for cluster_id, cluster_df in results.groupby("cluster"):
        sample = cluster_df.sample(n=min(gallery_size, len(cluster_df)), random_state=42)
        image_paths = [Path(path) for path in sample["image_path"] if Path(path).exists()]
        if not image_paths:
            continue

        columns = min(5, len(image_paths))
        rows = math.ceil(len(image_paths) / columns)
        fig, axes = plt.subplots(rows, columns, figsize=(columns * 2, rows * 2))
        axes = [axes] if len(image_paths) == 1 else axes.flatten()

        for ax, image_path in zip(axes, image_paths):
            ax.imshow(Image.open(image_path).convert("RGB"))
            ax.set_title(image_path.stem, fontsize=7)
            ax.axis("off")

        for ax in axes[len(image_paths):]:
            ax.axis("off")

        plt.tight_layout()
        plt.savefig(out_dir / f"cluster_{cluster_id}.png", dpi=200)
        plt.close(fig)


if __name__ == "__main__":
    main()
