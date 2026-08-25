from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp
import torch
import yaml
from sklearn.model_selection import train_test_split


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def row_normalize(features: sp.spmatrix | np.ndarray) -> sp.spmatrix | np.ndarray:
    rowsum = np.asarray(features.sum(axis=1)).reshape(-1)
    rowsum[rowsum == 0] = 1.0
    inv = 1.0 / rowsum
    if sp.issparse(features):
        return sp.diags(inv).dot(features)
    return features * inv[:, None]


def scipy_to_torch_dense(features: sp.spmatrix | np.ndarray) -> torch.Tensor:
    if sp.issparse(features):
        features = features.toarray()
    return torch.tensor(np.asarray(features), dtype=torch.float32).contiguous()


def edge_index_from_sparse(adj: sp.spmatrix, make_undirected: bool = True) -> torch.Tensor:
    if make_undirected:
        adj = adj + adj.T
    adj = adj.tocoo()
    edges = np.vstack([adj.row, adj.col])
    return torch.tensor(edges, dtype=torch.long).contiguous()


def stratified_masks(
    labels: np.ndarray | torch.Tensor,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    labels_np = labels.cpu().numpy() if isinstance(labels, torch.Tensor) else np.asarray(labels)
    indices = np.arange(labels_np.shape[0])
    stratify = labels_np if _can_stratify(labels_np) else None
    train_idx, rest_idx, train_y, rest_y = train_test_split(
        indices,
        labels_np,
        train_size=train_ratio,
        random_state=seed,
        shuffle=True,
        stratify=stratify,
    )
    rest_ratio = 1.0 - train_ratio
    test_ratio_within_rest = (1.0 - train_ratio - val_ratio) / rest_ratio
    rest_stratify = rest_y if _can_stratify(rest_y) else None
    val_idx, test_idx = train_test_split(
        rest_idx,
        test_size=test_ratio_within_rest,
        random_state=seed,
        shuffle=True,
        stratify=rest_stratify,
    )
    train_mask = torch.zeros(labels_np.shape[0], dtype=torch.bool)
    val_mask = torch.zeros(labels_np.shape[0], dtype=torch.bool)
    test_mask = torch.zeros(labels_np.shape[0], dtype=torch.bool)
    train_mask[torch.tensor(train_idx, dtype=torch.long)] = True
    val_mask[torch.tensor(val_idx, dtype=torch.long)] = True
    test_mask[torch.tensor(test_idx, dtype=torch.long)] = True
    return train_mask, val_mask, test_mask


def _can_stratify(labels: np.ndarray) -> bool:
    _, counts = np.unique(labels, return_counts=True)
    return bool(counts.size > 1 and counts.min() >= 2)
