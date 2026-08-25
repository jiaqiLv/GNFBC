from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def normalized_adjacency(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    row, col = edge_index
    device = edge_index.device
    self_loop = torch.arange(num_nodes, device=device)
    row = torch.cat([row, self_loop])
    col = torch.cat([col, self_loop])
    deg = torch.bincount(row, minlength=num_nodes).float().clamp_min(1.0)
    value = 1.0 / deg[row]
    return torch.sparse_coo_tensor(
        torch.stack([row, col]),
        value,
        (num_nodes, num_nodes),
        device=device,
        check_invariants=False,
    ).coalesce()


def dirichlet_feedback_coefficients(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    row, col = edge_index
    num_nodes = x.size(0)
    deg = torch.bincount(row, minlength=num_nodes).float().clamp_min(1.0)
    xi = x[row] / torch.sqrt(deg[row]).unsqueeze(-1)
    xj = x[col] / torch.sqrt(deg[col]).unsqueeze(-1)
    energy_per_edge = 0.25 * (xi - xj).pow(2).sum(dim=1)
    energy = torch.zeros(num_nodes, device=x.device).index_add(0, row, energy_per_edge)
    min_energy = energy.min()
    max_energy = energy.max()
    norm_energy = (energy - min_energy) / (max_energy - min_energy + eps)
    return (1.0 - norm_energy).clamp(0.0, 1.0)


class SharedSAGELayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.linear = nn.Linear(in_channels * 2, out_channels)

    def graph_aware(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        neigh = torch.sparse.mm(adj, x)
        return self.linear(torch.cat([x, neigh], dim=-1))

    def graph_agnostic(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(torch.cat([x, x], dim=-1))


class GNFBC(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        dropout: float,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")

        widths = [in_channels]
        if num_layers == 1:
            widths.append(out_channels)
        else:
            widths.extend([hidden_channels] * (num_layers - 1))
            widths.append(out_channels)

        self.layers = nn.ModuleList(
            SharedSAGELayer(widths[i], widths[i + 1]) for i in range(num_layers)
        )
        self.dropout = dropout

    def forward_train(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        beta: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        h = x
        corrected_logits = []
        beta_col = beta.unsqueeze(-1)
        for layer_idx, layer in enumerate(self.layers):
            aware = layer.graph_aware(h, adj)
            agnostic = layer.graph_agnostic(h)
            h = (1.0 - beta_col) * aware + beta_col * agnostic
            if layer_idx != len(self.layers) - 1:
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
            corrected_logits.append(h)
        return h, corrected_logits

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = x
        for layer_idx, layer in enumerate(self.layers):
            h = layer.graph_aware(h, adj)
            if layer_idx != len(self.layers) - 1:
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
        return h

    def forward_agnostic(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for layer_idx, layer in enumerate(self.layers):
            h = layer.graph_agnostic(h)
            if layer_idx != len(self.layers) - 1:
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
        return h
