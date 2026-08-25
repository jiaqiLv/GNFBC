from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class GraphData:
    name: str
    x: torch.Tensor
    edge_index: torch.Tensor
    y: torch.Tensor
    train_mask: torch.Tensor
    val_mask: torch.Tensor
    test_mask: torch.Tensor
    num_classes: int

    def to(self, device: torch.device | str) -> "GraphData":
        return GraphData(
            name=self.name,
            x=self.x.to(device),
            edge_index=self.edge_index.to(device),
            y=self.y.to(device),
            train_mask=self.train_mask.to(device),
            val_mask=self.val_mask.to(device),
            test_mask=self.test_mask.to(device),
            num_classes=self.num_classes,
        )
