from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from gnfbc.datasets import normalize_dataset_name
from gnfbc.train import train_once
from gnfbc.utils import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GNFBC experiments and write logs.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--datasets", nargs="+", default=["cora", "texas"])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--hidden-channels", type=int, default=128)
    parser.add_argument("--output-root", default="runs")
    parser.add_argument("--log-every", type=int, default=10)
    args = parser.parse_args()

    config = load_yaml(args.config)
    config["training"]["epochs"] = args.epochs
    config["runtime"]["device"] = args.device
    config["model"]["hidden_channels"] = args.hidden_channels

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for raw_name in args.datasets:
        dataset = normalize_dataset_name(raw_name)
        log_path = output_dir / f"{dataset}.log"
        print(f"Running {dataset}, log: {log_path}")
        with log_path.open("w", encoding="utf-8") as log:
            log.write(f"dataset: {dataset}\n")
            log.write(f"device_requested: {args.device}\n")
            log.write(f"cuda_available: {torch.cuda.is_available()}\n")
            if torch.cuda.is_available():
                log.write(f"cuda_device: {torch.cuda.get_device_name(0)}\n")
            log.write(f"epochs: {args.epochs}\n")
            log.write(f"hidden_channels: {args.hidden_channels}\n")
            log.write("\n")

            metrics = train_once(dataset, config, allow_download=False, log_every=0)
            for row in metrics["history"]:
                if row["epoch"] == 1 or row["epoch"] % args.log_every == 0:
                    log.write(
                        "epoch={epoch:.0f} loss={loss:.6f} ce={cross_entropy:.6f} "
                        "nf={negative_feedback_loss:.6f} train_acc={train_accuracy:.4f} "
                        "train_f1={train_f1_macro:.4f} val_acc={val_accuracy:.4f} "
                        "val_f1={val_f1_macro:.4f} val_auc={val_auc:.4f}\n".format(**row)
                    )
            log.write("\nfinal_metrics:\n")
            for key, value in metrics.items():
                if key != "history":
                    log.write(f"{key}: {value}\n")

        json_path = output_dir / f"{dataset}.json"
        with json_path.open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2, allow_nan=True)

        summary_rows.append(
            {
                "dataset": dataset,
                "accuracy": metrics["accuracy"],
                "f1_macro": metrics["f1_macro"],
                "auc": metrics["auc"],
                "best_val_accuracy": metrics["best_val_accuracy"],
                "best_epoch": metrics["best_epoch"],
                "elapsed_seconds": metrics["elapsed_seconds"],
                "nodes": metrics["num_nodes"],
                "edges": metrics["num_edges"],
            }
        )

    summary_path = output_dir / "summary.md"
    with summary_path.open("w", encoding="utf-8") as handle:
        handle.write("# GNFBC Experiment Summary\n\n")
        handle.write(f"- Run ID: `{run_id}`\n")
        handle.write(f"- Device requested: `{args.device}`\n")
        handle.write(f"- CUDA available: `{torch.cuda.is_available()}`\n")
        if torch.cuda.is_available():
            handle.write(f"- CUDA device: `{torch.cuda.get_device_name(0)}`\n")
        handle.write(f"- Epoch budget: `{args.epochs}`\n")
        handle.write(f"- Hidden channels: `{args.hidden_channels}`\n\n")
        handle.write("- Early stopping patience: `{}`\n".format(config["training"]["patience"]))
        handle.write(
            "- Train/validation/test split: `{} / {} / {}`\n\n".format(
                config["data"]["train_ratio"],
                config["data"]["val_ratio"],
                round(1 - config["data"]["train_ratio"] - config["data"]["val_ratio"], 6),
            )
        )
        handle.write("## Command\n\n")
        handle.write("```powershell\n")
        handle.write(
            ".\\.venv\\Scripts\\python scripts\\run_experiments.py "
            f"--datasets {' '.join(args.datasets)} --epochs {args.epochs} "
            f"--device {args.device} --hidden-channels {args.hidden_channels} "
            f"--log-every {args.log_every}\n"
        )
        handle.write("```\n\n")
        handle.write("| Dataset | Test Acc | Test F1 Macro | Test AUC | Best Val Acc | Best Epoch | Seconds | Nodes | Directed Edges |\n")
        handle.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in summary_rows:
            handle.write(
                f"| {row['dataset']} | {row['accuracy']:.4f} | {row['f1_macro']:.4f} | "
                f"{row['auc']:.4f} | {row['best_val_accuracy']:.4f} | "
                f"{row['best_epoch']:.0f} | {row['elapsed_seconds']:.2f} | "
                f"{row['nodes']:.0f} | {row['edges']:.0f} |\n"
            )
        handle.write("\n## Notes\n\n")
        handle.write("- Logs are written per dataset as `.log` files.\n")
        handle.write("- Full metric histories are stored as `.json` files.\n")
        handle.write("- Directed edge counts come from the internal undirected `edge_index` representation.\n")
        handle.write("- Single-run results are useful for pipeline validation, not paper-level comparison.\n")

    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
