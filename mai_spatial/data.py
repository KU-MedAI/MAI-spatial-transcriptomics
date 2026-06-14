"""Training dataset helpers."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

METADATA_COLUMNS = {
    "id",
    "path",
    "barcode",
    "tissue_index",
    "array_row",
    "array_col",
    "pxl_row",
    "pxl_col",
    "cluster",
    "fold",
}

SPOT_SORT_COLUMNS = ["tissue_index", "array_row", "array_col", "pxl_row", "pxl_col"]


def sort_spots(df: pd.DataFrame) -> pd.DataFrame:
    sort_columns = [column for column in SPOT_SORT_COLUMNS if column in df.columns]
    if not sort_columns:
        return df.reset_index(drop=True)
    return df.sort_values(sort_columns).reset_index(drop=True)


def detect_gene_columns(df: pd.DataFrame) -> list[str]:
    gene_columns = [
        column
        for column in df.columns
        if column not in METADATA_COLUMNS and pd.api.types.is_numeric_dtype(df[column])
    ]
    if not gene_columns:
        raise ValueError("No numeric gene-expression columns were found in the CSV.")
    return gene_columns


def read_expression_csv(path: str | Path) -> tuple[pd.DataFrame, list[str]]:
    df = sort_spots(pd.read_csv(path))
    return df, detect_gene_columns(df)


def split_train_valid(
    df: pd.DataFrame,
    seed: int,
    valid_size: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df, valid_df = train_test_split(
        df,
        test_size=valid_size,
        random_state=seed,
        shuffle=True,
    )
    return train_df.reset_index(drop=True), valid_df.reset_index(drop=True)


class GeneExpressionDataset(Dataset):
    """CSV-backed image-to-gene-expression dataset."""

    def __init__(
        self,
        df: pd.DataFrame,
        gene_columns: list[str],
        transform=None,
        has_labels: bool = True,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.gene_columns = gene_columns
        self.transform = transform
        self.has_labels = has_labels and all(column in df.columns for column in gene_columns)
        self.targets = (
            None
            if not self.has_labels
            else df[gene_columns].to_numpy(dtype=np.float32)
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int):
        image_path = self.df.loc[index, "path"]
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")

        if self.transform is not None:
            image = self.transform(image=image)["image"]

        if self.has_labels:
            return image, torch.from_numpy(self.targets[index])

        return image
