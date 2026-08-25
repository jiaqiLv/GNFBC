from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import time

import torch
import torch.nn.functional as F

from gnfbc.datasets import DATASETS, load_dataset, normalize_dataset_name
from gnfbc.metrics import classification_metrics
from gnfbc.models import GNFBC, dirichlet_feedback_coefficients, normalized_adjacency
from gnfbc.utils import load_yaml, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the GNFBC reproduction model.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--hidden-channels", type=int, default=None)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--feedback-loss-weight", type=float, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--train-ratio", type=float, default=None)
    parser.add_argument("--val-ratio", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def merged_config(args: argparse.Namespace) -> dict:
    config = load_yaml(args.config)
    if args.data_root is not None:
        config["data"]["root"] = args.data_root
    if args.train_ratio is not None:
        config["data"]["train_ratio"] = args.train_ratio
    if args.val_ratio is not None:
        config["data"]["val_ratio"] = args.val_ratio
    if args.seed is not None:
        config["data"]["seed"] = args.seed
    if args.hidden_channels is not None:
        config["model"]["hidden_channels"] = args.hidden_channels
    if args.num_layers is not None:
        config["model"]["num_layers"] = args.num_layers
    if args.dropout is not None:
        config["model"]["dropout"] = args.dropout
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.lr is not None:
        config["training"]["learning_rate"] = args.lr
    if args.weight_decay is not None:
        config["training"]["weight_decay"] = args.weight_decay
    if args.feedback_loss_weight is not None:
        config["training"]["feedback_loss_weight"] = args.feedback_loss_weight
    if args.patience is not None:
        config["training"]["patience"] = args.patience
    if args.device is not None:
        config["runtime"]["device"] = args.device
    return config


def negative_feedback_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    edge_index: torch.Tensor,
    beta: torch.Tensor,
) -> torch.Tensor:
    probs = F.softmax(logits, dim=-1)
    row, col = edge_index
    smooth_penalty = (probs[row] - probs[col]).pow(2).sum(dim=1)
    weighted_penalty = (beta[row] * smooth_penalty).mean()
    return weighted_penalty


def predict(
    model: GNFBC,
    graph,
    adj: torch.Tensor,
    beta: torch.Tensor,
    inference_mode: str,
) -> torch.Tensor:
    if inference_mode == "aware":
        return model(graph.x, adj)
    if inference_mode == "corrected":
        logits, _ = model.forward_train(graph.x, adj, beta)
        return logits
    if inference_mode == "agnostic":
        return model.forward_agnostic(graph.x)
    raise ValueError(f"Unknown inference mode: {inference_mode}")


def evaluate(
    model: GNFBC,
    graph,
    adj: torch.Tensor,
    beta: torch.Tensor,
    mask: torch.Tensor,
    inference_mode: str,
) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        logits = predict(model, graph, adj, beta, inference_mode)
    return classification_metrics(logits[mask], graph.y[mask])


def train_once(
    dataset_name: str,
    config: dict,
    allow_download: bool = True,
    log_every: int = 0,
) -> dict[str, float | list[dict[str, float]]]:
    seed = int(config["data"]["seed"])
    set_seed(seed)
    device_name = config["runtime"]["device"]
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)

    graph = load_dataset(
        dataset_name,
        root=config["data"]["root"],
        train_ratio=float(config["data"]["train_ratio"]),
        val_ratio=float(config["data"]["val_ratio"]),
        seed=seed,
        download=allow_download,
    ).to(device)
    adj = normalized_adjacency(graph.edge_index, graph.x.size(0))
    beta = dirichlet_feedback_coefficients(graph.x, graph.edge_index)

    model = GNFBC(
        in_channels=graph.x.size(1),
        hidden_channels=int(config["model"]["hidden_channels"]),
        out_channels=graph.num_classes,
        num_layers=int(config["model"]["num_layers"]),
        dropout=float(config["model"]["dropout"]),
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )

    best_state = deepcopy(model.state_dict())
    best_val = -1.0
    best_epoch = 0
    patience = int(config["training"]["patience"])
    feedback_weight = float(config["training"]["feedback_loss_weight"])
    aware_weight = float(config["training"].get("aware_loss_weight", 1.0))
    inference_mode = str(config.get("runtime", {}).get("inference_mode", "aware"))

    history: list[dict[str, float]] = []
    started_at = time.perf_counter()
    for epoch in range(1, int(config["training"]["epochs"]) + 1):
        model.train()
        optimizer.zero_grad()
        logits, _ = model.forward_train(graph.x, adj, beta)
        aware_logits = model(graph.x, adj)
        ce = F.cross_entropy(logits[graph.train_mask], graph.y[graph.train_mask])
        aware_ce = F.cross_entropy(aware_logits[graph.train_mask], graph.y[graph.train_mask])
        nf = negative_feedback_loss(logits, graph.y, graph.train_mask, graph.edge_index, beta)
        loss = ce + aware_weight * aware_ce + feedback_weight * nf
        loss.backward()
        optimizer.step()

        train_metrics = classification_metrics(logits[graph.train_mask], graph.y[graph.train_mask])
        val_metrics = evaluate(model, graph, adj, beta, graph.val_mask, inference_mode)
        val_score = val_metrics["accuracy"]
        row = {
            "epoch": float(epoch),
            "loss": float(loss.detach().cpu()),
            "cross_entropy": float(ce.detach().cpu()),
            "aware_cross_entropy": float(aware_ce.detach().cpu()),
            "negative_feedback_loss": float(nf.detach().cpu()),
            "train_accuracy": train_metrics["accuracy"],
            "train_f1_macro": train_metrics["f1_macro"],
            "train_auc": train_metrics["auc"],
            "val_accuracy": val_metrics["accuracy"],
            "val_f1_macro": val_metrics["f1_macro"],
            "val_auc": val_metrics["auc"],
        }
        history.append(row)
        if log_every and (epoch == 1 or epoch % log_every == 0):
            print(
                "epoch={epoch:.0f} loss={loss:.6f} train_acc={train_accuracy:.4f} "
                "val_acc={val_accuracy:.4f} val_f1={val_f1_macro:.4f}".format(**row),
                flush=True,
            )
        if val_score > best_val:
            best_val = val_score
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
        elif epoch - best_epoch >= patience:
            break

    model.load_state_dict(best_state)
    test_metrics = evaluate(model, graph, adj, beta, graph.test_mask, inference_mode)
    test_metrics["best_val_accuracy"] = best_val
    test_metrics["best_epoch"] = float(best_epoch)
    test_metrics["num_nodes"] = float(graph.x.size(0))
    test_metrics["num_edges"] = float(graph.edge_index.size(1))
    test_metrics["elapsed_seconds"] = float(time.perf_counter() - started_at)
    test_metrics["inference_mode"] = inference_mode
    test_metrics["history"] = history
    return test_metrics


def main() -> None:
    args = parse_args()
    config = merged_config(args)
    dataset = normalize_dataset_name(args.dataset)
    metrics = train_once(
        dataset,
        config,
        allow_download=not args.no_download,
        log_every=args.log_every,
    )
    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / f"{dataset}.json").open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2, allow_nan=True)
    print(f"Dataset: {dataset}")
    for key, value in metrics.items():
        if key == "history":
            continue
        if isinstance(value, (int, float)):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
