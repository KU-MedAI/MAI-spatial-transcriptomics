"""Visium slide loading and patch extraction."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, default_collate
from tqdm import tqdm

POSITION_COLUMNS = [
    "barcode",
    "in_tissue",
    "array_row",
    "array_col",
    "pxl_row",
    "pxl_col",
]


@dataclass(frozen=True)
class SlideConfig:
    tissue_index: int
    image_candidates: tuple[str, ...]
    spatial_position_file: str
    expression_file: str
    barcode_file: str


SLIDES = (
    SlideConfig(
        tissue_index=1,
        image_candidates=(
            "GSM7697868_GEX_C73_A1_Merged.tiff",
            "GSM7697868_GEX_C73_A1_Merged.tif",
            "GEX_C73_A1_Merged.tif",
        ),
        spatial_position_file="data/tissue_pos_matrices/tissue_positions_list_1.csv",
        expression_file="data/filtered_expression_matrices/1/harmony_matrix.npy",
        barcode_file="data/filtered_expression_matrices/1/barcodes.tsv",
    ),
    SlideConfig(
        tissue_index=2,
        image_candidates=(
            "GSM7697869_GEX_C73_B1_Merged.tiff",
            "GSM7697869_GEX_C73_B1_Merged.tif",
            "GEX_C73_B1_Merged.tif",
        ),
        spatial_position_file="data/tissue_pos_matrices/tissue_positions_list_2.csv",
        expression_file="data/filtered_expression_matrices/2/harmony_matrix.npy",
        barcode_file="data/filtered_expression_matrices/2/barcodes.tsv",
    ),
    SlideConfig(
        tissue_index=3,
        image_candidates=(
            "GSM7697870_GEX_C73_C1_Merged.tiff",
            "GSM7697870_GEX_C73_C1_Merged.tif",
            "GEX_C73_C1_Merged.tif",
        ),
        spatial_position_file="data/tissue_pos_matrices/tissue_positions_list_3.csv",
        expression_file="data/filtered_expression_matrices/3/harmony_matrix.npy",
        barcode_file="data/filtered_expression_matrices/3/barcodes.tsv",
    ),
    SlideConfig(
        tissue_index=4,
        image_candidates=(
            "GSM7697871_GEX_C73_D1_Merged.tiff",
            "GSM7697871_GEX_C73_D1_Merged.tif",
            "GEX_C73_D1_Merged.tif",
        ),
        spatial_position_file="data/tissue_pos_matrices/tissue_positions_list_4.csv",
        expression_file="data/filtered_expression_matrices/4/harmony_matrix.npy",
        barcode_file="data/filtered_expression_matrices/4/barcodes.tsv",
    ),
)


class VisiumPatchDataset(Dataset):
    """Load one Visium slide and return image patches aligned to expression rows."""

    def __init__(
        self,
        image_path: str | Path,
        spatial_position_path: str | Path,
        expression_path: str | Path,
        barcode_path: str | Path,
        patch_size: int = 224,
        in_tissue_only: bool = True,
    ) -> None:
        self.image_path = Path(image_path)
        self.patch_size = int(patch_size)
        self.half_patch = self.patch_size // 2

        self.image = cv2.imread(str(self.image_path), cv2.IMREAD_COLOR)
        if self.image is None:
            raise FileNotFoundError(f"Could not load image: {self.image_path}")

        all_barcodes = pd.read_csv(barcode_path, sep="\t", header=None)[0].astype(str)
        positions = pd.read_csv(
            spatial_position_path,
            header=None,
            names=POSITION_COLUMNS,
        )
        positions["barcode"] = positions["barcode"].astype(str)
        positions = positions.set_index("barcode").reindex(all_barcodes)

        missing_positions = positions.index[positions["array_row"].isna()].tolist()
        if missing_positions:
            preview = ", ".join(missing_positions[:5])
            raise ValueError(f"{len(missing_positions)} barcodes are missing positions: {preview}")

        expression = self._align_expression(np.load(expression_path, mmap_mode="r"), len(all_barcodes))

        if in_tissue_only:
            keep_mask = positions["in_tissue"].astype(int).to_numpy() == 1
            positions = positions.iloc[keep_mask].reset_index()
            barcodes = all_barcodes.iloc[keep_mask].reset_index(drop=True)
            expression = expression[keep_mask]
        else:
            positions = positions.reset_index()
            barcodes = all_barcodes.reset_index(drop=True)

        if expression.shape[0] != len(barcodes):
            raise ValueError(
                "Expression rows do not match barcode rows after filtering: "
                f"{expression.shape[0]} vs {len(barcodes)}"
            )

        self.barcodes = barcodes.to_numpy()
        self.positions = positions.reset_index(drop=True)
        self.expression = expression

    def __len__(self) -> int:
        return len(self.barcodes)

    def __getitem__(self, index: int) -> dict[str, object]:
        position = self.positions.iloc[index]
        center_row = int(position["pxl_row"])
        center_col = int(position["pxl_col"])
        expression = np.asarray(self.expression[index], dtype=np.float32).copy()

        return {
            "image_patch": self._extract_patch(center_row, center_col),
            "barcode": str(self.barcodes[index]),
            "expression": torch.from_numpy(expression),
            "array_indices": torch.tensor(
                [int(position["array_row"]), int(position["array_col"])],
                dtype=torch.long,
            ),
            "pixel_coordinates": torch.tensor([center_row, center_col], dtype=torch.long),
        }

    def _extract_patch(self, center_row: int, center_col: int) -> np.ndarray:
        top = center_row - self.half_patch
        bottom = top + self.patch_size
        left = center_col - self.half_patch
        right = left + self.patch_size

        pad_top = max(0, -top)
        pad_bottom = max(0, bottom - self.image.shape[0])
        pad_left = max(0, -left)
        pad_right = max(0, right - self.image.shape[1])

        if any((pad_top, pad_bottom, pad_left, pad_right)):
            image = cv2.copyMakeBorder(
                self.image,
                pad_top,
                pad_bottom,
                pad_left,
                pad_right,
                borderType=cv2.BORDER_CONSTANT,
                value=(255, 255, 255),
            )
            top += pad_top
            bottom += pad_top
            left += pad_left
            right += pad_left
        else:
            image = self.image

        return image[top:bottom, left:right].copy()

    @staticmethod
    def _align_expression(expression: np.ndarray, n_barcodes: int) -> np.ndarray:
        if expression.shape[0] == n_barcodes:
            return expression
        if expression.ndim == 2 and expression.shape[1] == n_barcodes:
            return expression.T
        raise ValueError(
            "Expression matrix does not match barcode count: "
            f"shape={expression.shape}, barcodes={n_barcodes}"
        )


def patch_collate_fn(batch):
    list_keys = {"image_patch", "barcode"}
    stacked_pair_keys = {"array_indices", "pixel_coordinates"}
    collated = {
        key: default_collate([item[key] for item in batch])
        for key in batch[0]
        if key not in list_keys and key not in stacked_pair_keys
    }
    collated["image_patch"] = [item["image_patch"] for item in batch]
    collated["barcode"] = [item["barcode"] for item in batch]
    collated["array_indices"] = torch.stack([item["array_indices"] for item in batch], dim=1)
    collated["pixel_coordinates"] = torch.stack(
        [item["pixel_coordinates"] for item in batch],
        dim=1,
    )
    return collated


def resolve_slide_image(data_root: Path, slide: SlideConfig) -> Path:
    image_dir = data_root / "image"
    for candidate in slide.image_candidates:
        path = image_dir / candidate
        if path.exists():
            return path
    return image_dir / slide.image_candidates[0]


def load_gene_columns(data_root: str | Path, fallback_count: int = 3467) -> list[str]:
    data_root = Path(data_root)
    feature_path = data_root / "data/filtered_expression_matrices/1/features.tsv"
    hvg_path = data_root / "data/filtered_expression_matrices/hvg_union.npy"

    try:
        all_gene_names = pd.read_csv(feature_path, sep="\t", header=None)[1].astype(str).values
        hvg_mask = np.load(hvg_path)
        return list(all_gene_names[hvg_mask])
    except Exception as exc:
        print(f"[WARN] Falling back to generated gene names: {exc}")
        return [f"gene_{index:04d}" for index in range(fallback_count)]


def create_patch_dataset_csv(
    data_root: str | Path,
    out_csv_path: str | Path,
    out_image_dir: str | Path,
    batch_size: int = 32,
    num_workers: int = 4,
    patch_size: int = 224,
) -> None:
    data_root = Path(data_root)
    out_csv_path = Path(out_csv_path)
    out_image_dir = Path(out_image_dir)
    out_image_dir.mkdir(parents=True, exist_ok=True)

    gene_columns = load_gene_columns(data_root)
    rows = []
    global_index = 0

    for slide in SLIDES:
        dataset = VisiumPatchDataset(
            image_path=resolve_slide_image(data_root, slide),
            spatial_position_path=data_root / slide.spatial_position_file,
            expression_path=data_root / slide.expression_file,
            barcode_path=data_root / slide.barcode_file,
            patch_size=patch_size,
        )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=patch_collate_fn,
        )

        for batch in tqdm(loader, desc=f"Processing tissue {slide.tissue_index}", unit="batch"):
            patches = batch["image_patch"]
            expressions = batch["expression"].cpu().numpy()
            array_indices = batch["array_indices"]
            pixel_coordinates = batch["pixel_coordinates"]

            if expressions.shape[1] != len(gene_columns):
                raise ValueError(
                    "Expression dimension does not match gene columns: "
                    f"{expressions.shape[1]} vs {len(gene_columns)}"
                )

            for index, barcode in enumerate(batch["barcode"]):
                sample_id = f"TRAIN_{global_index:05d}"
                image_path = out_image_dir / f"{sample_id}.png"
                cv2.imwrite(str(image_path), patches[index])

                row = {
                    "id": sample_id,
                    "path": str(image_path),
                    "barcode": barcode,
                    "tissue_index": slide.tissue_index,
                    "array_row": int(array_indices[0][index]),
                    "array_col": int(array_indices[1][index]),
                    "pxl_row": int(pixel_coordinates[0][index]),
                    "pxl_col": int(pixel_coordinates[1][index]),
                }
                row.update(
                    {
                        gene: float(value)
                        for gene, value in zip(gene_columns, expressions[index])
                    }
                )
                rows.append(row)
                global_index += 1

    metadata_columns = [
        "id",
        "path",
        "barcode",
        "tissue_index",
        "array_row",
        "array_col",
        "pxl_row",
        "pxl_col",
    ]
    dataframe = pd.DataFrame(rows)[metadata_columns + gene_columns]
    dataframe = dataframe.sort_values(
        ["tissue_index", "array_row", "array_col", "pxl_row", "pxl_col"]
    ).reset_index(drop=True)
    dataframe.to_csv(out_csv_path, index=False)

    metadata = {
        "data_root": str(data_root),
        "out_csv": str(out_csv_path),
        "out_image_dir": str(out_image_dir),
        "num_rows": int(len(dataframe)),
        "num_genes": int(len(gene_columns)),
        "patch_size": int(patch_size),
    }
    metadata_path = out_csv_path.with_name(f"{out_csv_path.stem}_meta.json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Saved dataset CSV: {out_csv_path}")
    print(f"Saved patches: {out_image_dir}")
    print(f"Saved metadata: {metadata_path}")
