"""Evaluation metrics."""

from __future__ import annotations

import numpy as np

from mai_spatial.markers import MARKER_GROUPS


def grouped_pearson_corr(
    pred: np.ndarray,
    true: np.ndarray,
    gene_columns: list[str],
    gene_groups: dict[str, list[str]] | None = None,
) -> dict[str, float]:
    """Average per-gene Pearson correlation for each marker group."""
    groups = MARKER_GROUPS if gene_groups is None else gene_groups
    column_index = {gene: index for index, gene in enumerate(gene_columns)}
    scores: dict[str, float] = {}

    for group_name, genes in groups.items():
        selected = [column_index[gene] for gene in genes if gene in column_index]
        if not selected:
            scores[group_name] = float("nan")
            continue

        group_pred = pred[:, selected]
        group_true = true[:, selected]
        correlations = []

        for column in range(group_pred.shape[1]):
            x = group_pred[:, column]
            y = group_true[:, column]
            if np.std(x) == 0 or np.std(y) == 0:
                correlations.append(0.0)
                continue

            corr = np.corrcoef(x, y)[0, 1]
            correlations.append(0.0 if not np.isfinite(corr) else float(corr))

        scores[group_name] = float(np.mean(correlations))

    return scores
