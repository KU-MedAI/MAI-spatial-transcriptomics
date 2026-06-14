"""Feature extraction and clustering helpers."""

from __future__ import annotations

import gc
from itertools import product
from pathlib import Path

import hdbscan
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torchvision.models as tv_models
import torchvision.transforms as tv_transforms
import umap
from PIL import Image
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def create_positional_features(df: pd.DataFrame, pos_dim: int = 64) -> np.ndarray:
    """Create 2D sinusoidal positional encodings per tissue section."""
    if pos_dim % 4 != 0:
        raise ValueError("pos_dim must be divisible by 4.")

    features = []
    for _, tissue_data in df.groupby("tissue_index", sort=False):
        x = tissue_data["array_row"].to_numpy()
        y = tissue_data["array_col"].to_numpy()

        x_range = x.max() - x.min()
        y_range = y.max() - y.min()
        x_norm = np.zeros_like(x, dtype=float) if x_range == 0 else (x - x.min()) / x_range
        y_norm = np.zeros_like(y, dtype=float) if y_range == 0 else (y - y.min()) / y_range
        features.append(_positional_encoding_2d(x_norm, y_norm, dim=pos_dim))

    return np.concatenate(features, axis=0)


def _positional_encoding_2d(x: np.ndarray, y: np.ndarray, dim: int, max_len: int = 10000):
    encoding = np.zeros((len(x), dim), dtype=np.float32)
    quarter = dim // 4
    div_term = np.exp(np.arange(0, quarter) * -(np.log(max_len) / quarter))
    encoding[:, 0::4] = np.sin(x[:, None] * div_term)
    encoding[:, 1::4] = np.cos(x[:, None] * div_term)
    encoding[:, 2::4] = np.sin(y[:, None] * div_term)
    encoding[:, 3::4] = np.cos(y[:, None] * div_term)
    return encoding


class PatchImageDataset(Dataset):
    """Load extracted patch PNGs for feature extraction."""

    def __init__(self, df: pd.DataFrame, image_dir: str | Path | None = None) -> None:
        self.df = df.reset_index(drop=True)
        self.image_dir = Path(image_dir) if image_dir is not None else None
        self.transform = tv_transforms.Compose(
            [
                tv_transforms.Resize((224, 224)),
                tv_transforms.ToTensor(),
                tv_transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int):
        row = self.df.iloc[index]
        path = self._resolve_path(row)
        image = Image.open(path).convert("RGB")
        return self.transform(image), path.name, str(path)

    def _resolve_path(self, row) -> Path:
        if "path" in row and pd.notna(row["path"]):
            return Path(row["path"])
        if self.image_dir is None:
            raise ValueError("image_dir is required when the CSV has no path column.")
        return self.image_dir / f"{row['id']}.png"


def load_histology_resnet18(
    model_path: str | Path = "weights/tenpercent_resnet18.ckpt",
    device: str | torch.device = "cuda",
) -> nn.Module:
    """Load a histology-pretrained ResNet18 checkpoint, with ImageNet fallback."""
    model = tv_models.resnet18(weights=None)

    try:
        state = torch.load(model_path, map_location=device)
        state_dict = state.get("state_dict", state)
        cleaned = {
            key.replace("model.", "").replace("resnet.", ""): value
            for key, value in state_dict.items()
        }
        model.load_state_dict(
            {**model.state_dict(), **{k: v for k, v in cleaned.items() if k in model.state_dict()}}
        )
    except Exception as exc:
        print(f"[WARN] Could not load {model_path}: {exc}")
        print("[WARN] Falling back to ImageNet ResNet18 weights.")
        model = tv_models.resnet18(weights=tv_models.ResNet18_Weights.DEFAULT)

    model.fc = nn.Identity()
    model = model.to(device)
    model.eval()
    return model


class HistologyFeatureExtractor:
    def __init__(
        self,
        model_path: str | Path = "weights/tenpercent_resnet18.ckpt",
        device: str | torch.device | None = None,
    ) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = load_histology_resnet18(model_path, self.device)

    @torch.no_grad()
    def extract(self, loader: DataLoader) -> tuple[np.ndarray, list[str], list[str]]:
        features = []
        names = []
        paths = []

        for images, image_names, image_paths in tqdm(loader, desc="Extracting features"):
            images = images.to(self.device)
            batch_features = self.model(images)
            features.append(batch_features.cpu().numpy())
            names.extend(image_names)
            paths.extend(image_paths)
            if len(features) % 10 == 0 and self.device.type == "cuda":
                gc.collect()
                torch.cuda.empty_cache()

        return np.concatenate(features, axis=0), names, paths


def fuse_features(
    histology_features: np.ndarray,
    positional_features: np.ndarray,
    histology_weight: float = 1.0,
    positional_weight: float = 100.0,
) -> np.ndarray:
    histology_norm = histology_features / np.linalg.norm(
        histology_features,
        axis=1,
        keepdims=True,
    )
    positional_norm = positional_features / np.linalg.norm(
        positional_features,
        axis=1,
        keepdims=True,
    )
    return np.concatenate(
        [
            histology_norm * histology_weight,
            positional_norm * positional_weight,
        ],
        axis=1,
    )


def evaluate_clusters(features: np.ndarray, clusters: np.ndarray) -> dict[str, float]:
    valid_mask = clusters != -1
    if np.sum(valid_mask) < 2:
        return {"silhouette": -1.0, "calinski": 0.0, "davies": float("inf"), "n_clusters": 0}

    valid_features = features[valid_mask]
    valid_clusters = clusters[valid_mask]
    n_clusters = len(set(valid_clusters))
    if n_clusters <= 1:
        return {
            "silhouette": -1.0,
            "calinski": 0.0,
            "davies": float("inf"),
            "n_clusters": n_clusters,
        }

    return {
        "silhouette": float(silhouette_score(valid_features, valid_clusters)),
        "calinski": float(calinski_harabasz_score(valid_features, valid_clusters)),
        "davies": float(davies_bouldin_score(valid_features, valid_clusters)),
        "n_clusters": int(n_clusters),
    }


def cluster_features(features: np.ndarray):
    scaled = StandardScaler().fit_transform(features)
    embedding = umap.UMAP(
        n_neighbors=10,
        min_dist=0.001,
        n_components=2,
        random_state=42,
        metric="euclidean",
    ).fit_transform(scaled)
    clusters = hdbscan.HDBSCAN(
        min_cluster_size=70,
        min_samples=3,
        cluster_selection_epsilon=0.05,
        metric="euclidean",
    ).fit_predict(embedding)
    return embedding, clusters, evaluate_clusters(embedding, clusters)


def run_weight_search(
    data_csv: str | Path = "dataset.csv",
    image_dir: str | Path | None = None,
    out_dir: str | Path = "runs/cluster_weight_search",
    tissues: tuple[int, ...] = (1, 2, 4),
    model_path: str | Path = "weights/tenpercent_resnet18.ckpt",
    quick: bool = False,
) -> pd.DataFrame:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_csv)
    df = df[df["tissue_index"].isin(tissues)].reset_index(drop=True)
    dataset = PatchImageDataset(df, image_dir=image_dir)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2)

    positional_features = create_positional_features(df)
    histology_features, names, paths = HistologyFeatureExtractor(model_path=model_path).extract(loader)

    histology_weights = [0.5, 1.0, 2.0, 5.0] if quick else [0.1, 0.5, 1.0, 2.0, 5.0]
    positional_weights = [50.0, 100.0, 500.0] if quick else [1.0, 10.0, 50.0, 100.0, 500.0]

    all_results = []
    for histology_weight, positional_weight in tqdm(
        list(product(histology_weights, positional_weights)),
        desc="Searching weights",
    ):
        fused = fuse_features(
            histology_features,
            positional_features,
            histology_weight=histology_weight,
            positional_weight=positional_weight,
        )
        embedding, clusters, metrics = cluster_features(fused)
        all_results.append(
            {
                "histology_weight": histology_weight,
                "positional_weight": positional_weight,
                "embedding": embedding,
                "clusters": clusters,
                **metrics,
            }
        )

    summary = pd.DataFrame(
        [
            {key: value for key, value in result.items() if key not in {"embedding", "clusters"}}
            for result in all_results
        ]
    ).sort_values("silhouette", ascending=False)
    summary.to_csv(out_dir / "cluster_weight_summary.csv", index=False)

    best = max(all_results, key=lambda result: result["silhouette"])
    result_df = pd.DataFrame(
        {
            "image_name": names,
            "image_path": paths,
            "cluster": best["clusters"],
            "array_row": df["array_row"].values,
            "array_col": df["array_col"].values,
            "tissue_index": df["tissue_index"].values,
            "umap_x": best["embedding"][:, 0],
            "umap_y": best["embedding"][:, 1],
        }
    )
    result_df.to_csv(out_dir / "cluster_assignments.csv", index=False)
    plot_cluster_embedding(result_df, out_dir / "cluster_embedding.png")

    print(summary.head(10))
    print(f"Saved: {out_dir / 'cluster_weight_summary.csv'}")
    print(f"Saved: {out_dir / 'cluster_assignments.csv'}")
    return summary


def plot_cluster_embedding(results: pd.DataFrame, out_path: str | Path) -> None:
    plt.figure(figsize=(8, 6))
    noise = results["cluster"] == -1
    if noise.any():
        plt.scatter(results.loc[noise, "umap_x"], results.loc[noise, "umap_y"], c="lightgray", s=8)
    clustered = ~noise
    if clustered.any():
        plt.scatter(
            results.loc[clustered, "umap_x"],
            results.loc[clustered, "umap_y"],
            c=results.loc[clustered, "cluster"],
            cmap="tab20",
            s=10,
        )
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_tissue_distribution(results: pd.DataFrame, out_path: str | Path) -> None:
    counts = pd.crosstab(results["cluster"], results["tissue_index"])
    plt.figure(figsize=(10, 6))
    sns.heatmap(counts, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Tissue")
    plt.ylabel("Cluster")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
