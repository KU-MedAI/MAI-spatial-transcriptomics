# MAI Spatial Transcriptomics

[![Paper](https://img.shields.io/badge/Paper-BMC%20Bioinformatics-blue)](https://doi.org/10.1186/s12859-026-06447-7)
[![Backbone](https://img.shields.io/badge/Backbone-EfficientNet--B0-green)]()
[![Task](https://img.shields.io/badge/Task-Spatial%20Gene%20Expression%20Prediction-orange)]()

Official implementation of:

> **From histology to spatial transcriptomics: establishing a lightweight single-patch baseline**  
> *BMC Bioinformatics*, 2026  
> DOI: https://doi.org/10.1186/s12859-026-06447-7

This repository trains lightweight single-patch models that predict spot-level gene expression from H&E histology patches. The main baseline uses EfficientNet-B0, controlled random seeds, fixed train/validation splits, and morphology-preserving augmentation.

## Highlights

- Single-spot prediction without neighboring spot context.
- EfficientNet-B0 baseline with a compact regression head.
- Morphology-preserving augmentation (MPA) for histology patches.
- Reusable experiment code under `mai_spatial/`.
- Clean command-line entry points under `scripts/`.

## Repository Layout

```text
MAI-spatial-transcriptomics/
├── mai_spatial/                 # Reusable package code
│   ├── data.py                   # Training CSV datasets and gene-column helpers
│   ├── engine.py                 # Train/eval/inference loops
│   ├── experiments.py            # Shared experiment runners
│   ├── markers.py                # HEG/HVG/MG marker groups
│   ├── metrics.py                # Grouped Pearson correlation
│   ├── models.py                 # Backbones and regression heads
│   ├── transforms.py             # Augmentation presets
│   └── visium.py                 # Visium slide loading and patch extraction
├── scripts/
│   ├── prepare_dataset.py        # Build patch PNGs and dataset.csv
│   ├── train_baseline.py         # EfficientNet-B0 baseline
│   ├── compare_backbones.py      # Backbone comparison
│   ├── compare_augmentations.py  # MPA/ST-Net/no-augmentation comparison
│   ├── search_finetuning.py      # 5-fold fine-tuning grid search
│   ├── search_cluster_weights.py # Histology/position cluster-weight search
│   └── export_cluster_samples.py # Cluster visualization helper
├── notebooks/                    # Exploratory notebooks
├── GSE240429_data/               # Preprocessed matrices and tissue positions
├── requirements.txt
└── README.md
```

## Setup

```bash
git clone https://github.com/KU-MedAI/MAI-spatial-transcriptomics.git
cd MAI-spatial-transcriptomics
pip install -r requirements.txt
```

The repository includes preprocessed expression matrices and tissue-position files. Full-size H&E images are not included because they are large. Download them from [NCBI GEO GSE240429](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE240429) and place them in:

```text
GSE240429_data/image/
```

Accepted image names include the GEO-prefixed names used by the original scripts and the shorter names listed in `GSE240429_data/image/full_sized_images.txt`.

## Data Preparation

Create patch PNGs and a single training CSV:

```bash
python -m scripts.prepare_dataset \
  --data-root GSE240429_data \
  --out-csv dataset.csv \
  --out-image-dir data/patches
```

The command writes:

- `dataset.csv`: metadata plus gene-expression targets.
- `data/patches/*.png`: extracted spot-centered histology patches.
- `dataset_meta.json`: dataset-generation metadata.

## Training

Train the EfficientNet-B0 baseline with the default 80/20 split:

```bash
python -m scripts.train_baseline \
  --data-csv dataset.csv \
  --out-dir runs/effb0_baseline
```

Useful options:

```bash
python -m scripts.train_baseline \
  --data-csv dataset.csv \
  --epochs 30 \
  --batch-size 32 \
  --learning-rate 3e-4 \
  --augmentation mpa
```

If you have a separate test CSV:

```bash
python -m scripts.train_baseline \
  --data-csv dataset.csv \
  --test-csv test.csv \
  --out-dir runs/effb0_baseline
```

## Experiment Scripts

Backbone comparison:

```bash
python -m scripts.compare_backbones \
  --data-csv dataset.csv \
  --out-dir runs/backbone_comparison
```

Augmentation comparison:

```bash
python -m scripts.compare_augmentations \
  --data-csv dataset.csv \
  --augmentations mpa,stnet,none \
  --out-dir runs/augmentation_comparison
```

Fine-tuning grid search:

```bash
python -m scripts.search_finetuning \
  --data-csv dataset.csv \
  --hidden-grid 1024,3000,4096 \
  --unfreeze-grid=-9,0 \
  --out-dir runs/finetuning_search
```

Cluster-weight exploration:

```bash
python -m scripts.search_cluster_weights \
  --data-csv dataset.csv \
  --out-dir runs/cluster_weight_search
```

Export cluster sample folders and galleries:

```bash
python -m scripts.export_cluster_samples \
  --cluster-csv runs/cluster_weight_search/cluster_assignments.csv \
  --out-dir runs/cluster_samples
```

## Outputs

Training and experiment outputs are written under `runs/` by default. Generated patches and model checkpoints are ignored by Git.

## Citation

```bibtex
@article{jang2026histology,
  title={From histology to spatial transcriptomics: establishing a lightweight single-patch baseline},
  author={Jang, Hyungyum and Shin, Hyunsoo and Lee, Hawon and Jang, Yena and Jung, Sunghoon and Jeon, Minji},
  journal={BMC Bioinformatics},
  year={2026},
  publisher={Springer}
}
```

## License

This project is licensed under the Apache-2.0 license.
