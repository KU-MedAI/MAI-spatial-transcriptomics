"""Create patch images and a training CSV from GSE240429 Visium files."""

from __future__ import annotations

import argparse

from mai_spatial.visium import create_patch_dataset_csv


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="GSE240429_data")
    parser.add_argument("--out-csv", default="dataset.csv")
    parser.add_argument("--out-image-dir", default="data/patches")
    parser.add_argument("--patch-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    create_patch_dataset_csv(
        data_root=args.data_root,
        out_csv_path=args.out_csv,
        out_image_dir=args.out_image_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        patch_size=args.patch_size,
    )


if __name__ == "__main__":
    main()
