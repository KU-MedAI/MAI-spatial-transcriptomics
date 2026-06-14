"""Search histology/position feature weights for clustering."""

from __future__ import annotations

import argparse

from mai_spatial.clustering import run_weight_search


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-csv", default="dataset.csv")
    parser.add_argument("--image-dir", default=None)
    parser.add_argument("--out-dir", default="runs/cluster_weight_search")
    parser.add_argument("--model-path", default="weights/tenpercent_resnet18.ckpt")
    parser.add_argument("--tissues", default="1,2,4")
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tissues = tuple(int(item.strip()) for item in args.tissues.split(",") if item.strip())
    run_weight_search(
        data_csv=args.data_csv,
        image_dir=args.image_dir,
        out_dir=args.out_dir,
        tissues=tissues,
        model_path=args.model_path,
        quick=args.quick,
    )


if __name__ == "__main__":
    main()
