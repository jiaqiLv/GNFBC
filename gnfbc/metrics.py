from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def classification_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    pred = logits.argmax(dim=-1)
    y_true = labels.detach().cpu().numpy()
    y_pred = pred.detach().cpu().numpy()
    result = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    probs = F.softmax(logits, dim=-1).detach().cpu().numpy()
    try:
        if probs.shape[1] == 2:
            result["auc"] = float(roc_auc_score(y_true, probs[:, 1]))
        else:
            result["auc"] = float(roc_auc_score(y_true, probs, multi_class="ovr"))
    except ValueError:
        result["auc"] = float("nan")
    return result
