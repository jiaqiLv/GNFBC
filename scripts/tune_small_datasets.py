from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from gnfbc.datasets import normalize_dataset_name
from gnfbc.train import train_once
from gnfbc.utils import load_yaml


PAPER_TARGETS = {
    "cora": {"accuracy": 0.8656},
    "citeseer": {"accuracy": 0.7391},
    "pubmed": {"accuracy": 0.8930},
    "computers": {"accuracy": 0.8043},
    "photo": {"accuracy": 0.8954},
    "chameleon": {"accuracy": 0.6742},
    "squirrel": {"accuracy": 0.5931},
    "texas": {"accuracy": 0.8990},
    "cornell": {"accuracy": 0.8889},
}


DEFAULT_DATASETS = [
    "cora",
    "citeseer",
    "pubmed",
    "chameleon",
    "squirrel",
    "texas",
    "cornell",
    "wisconsin",
    "washington",
]


TRIALS = [
    {
        "name": "base",
        "hidden_channels": 128,
        "num_layers": 2,
        "dropout": 0.5,
        "learning_rate": 0.001,
        "weight_decay": 0.0005,
        "feedback_loss_weight": 0.1,
        "aware_loss_weight": 1.0,
        "inference_mode": "aware",
    },
    {
        "name": "low_dropout",
        "hidden_channels": 128,
        "num_layers": 2,
        "dropout": 0.2,
        "learning_rate": 0.005,
        "weight_decay": 0.0005,
        "feedback_loss_weight": 0.05,
        "aware_loss_weight": 1.0,
        "inference_mode": "aware",
    },
    {
        "name": "wide",
        "hidden_channels": 256,
        "num_layers": 2,
        "dropout": 0.4,
        "learning_rate": 0.003,
        "weight_decay": 0.0005,
        "feedback_loss_weight": 0.05,
        "aware_loss_weight": 1.0,
        "inference_mode": "aware",
    },
    {
        "name": "deeper",
        "hidden_channels": 128,
        "num_layers": 3,
        "dropout": 0.3,
        "learning_rate": 0.003,
        "weight_decay": 0.001,
        "feedback_loss_weight": 0.05,
        "aware_loss_weight": 1.0,
        "inference_mode": "aware",
    },
    {
        "name": "shallow_fast",
        "hidden_channels": 128,
        "num_layers": 1,
        "dropout": 0.0,
        "learning_rate": 0.01,
        "weight_decay": 0.0005,
        "feedback_loss_weight": 0.01,
        "aware_loss_weight": 1.0,
        "inference_mode": "aware",
    },
    {
        "name": "shallow_regularized",
        "hidden_channels": 256,
        "num_layers": 1,
        "dropout": 0.3,
        "learning_rate": 0.01,
        "weight_decay": 0.005,
        "feedback_loss_weight": 0.01,
        "aware_loss_weight": 1.0,
        "inference_mode": "aware",
    },
    {
        "name": "no_feedback_sage",
        "hidden_channels": 128,
        "num_layers": 2,
        "dropout": 0.2,
        "learning_rate": 0.005,
        "weight_decay": 0.0005,
        "feedback_loss_weight": 0.0,
        "aware_loss_weight": 1.0,
        "inference_mode": "aware",
    },
    {
        "name": "corrected_low_dropout",
        "hidden_channels": 128,
        "num_layers": 2,
        "dropout": 0.2,
        "learning_rate": 0.005,
        "weight_decay": 0.0005,
        "feedback_loss_weight": 0.05,
        "aware_loss_weight": 1.0,
        "inference_mode": "corrected",
    },
    {
        "name": "agnostic_fast",
        "hidden_channels": 128,
        "num_layers": 2,
        "dropout": 0.2,
        "learning_rate": 0.005,
        "weight_decay": 0.0005,
        "feedback_loss_weight": 0.0,
        "aware_loss_weight": 0.0,
        "inference_mode": "agnostic",
    },
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune small GNFBC datasets against paper targets.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=60)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-root", default="runs")
    parser.add_argument("--max-trials", type=int, default=len(TRIALS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[2])
    parser.add_argument("--trial-names", nargs="*", default=None)
    args = parser.parse_args()

    base_config = load_yaml(args.config)
    base_config["training"]["epochs"] = args.epochs
    base_config["training"]["patience"] = args.patience
    base_config["runtime"]["device"] = args.device

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_tune")
    output_dir = Path(args.output_root) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []
    best_rows: list[dict[str, Any]] = []
    trials = TRIALS[: args.max_trials]
    if args.trial_names:
        requested = set(args.trial_names)
        trials = [trial for trial in TRIALS if trial["name"] in requested]
        missing = requested - {trial["name"] for trial in trials}
        if missing:
            raise SystemExit(f"Unknown trial names: {sorted(missing)}")

    for raw_dataset in args.datasets:
        dataset = normalize_dataset_name(raw_dataset)
        dataset_dir = output_dir / dataset
        dataset_dir.mkdir(parents=True, exist_ok=True)
        best: dict[str, Any] | None = None

        for trial in trials:
            for seed in args.seeds:
                run_trial(dataset, trial, seed, base_config, dataset_dir, args, all_rows)
                if all_rows[-1]["dataset"] == dataset and (
                    best is None or all_rows[-1]["accuracy"] > best["accuracy"]
                ):
                    best = all_rows[-1]

        if best is not None:
            best_rows.append(best)

    write_summary(output_dir, all_rows, best_rows, args, trials)
    print(f"Summary: {output_dir / 'summary.md'}")


def run_trial(
    dataset: str,
    trial: dict[str, Any],
    seed: int,
    base_config: dict[str, Any],
    dataset_dir: Path,
    args: argparse.Namespace,
    all_rows: list[dict[str, Any]],
) -> None:
    config = json.loads(json.dumps(base_config))
    config["data"]["seed"] = seed
    config["model"]["hidden_channels"] = trial["hidden_channels"]
    config["model"]["num_layers"] = trial["num_layers"]
    config["model"]["dropout"] = trial["dropout"]
    config["training"]["learning_rate"] = trial["learning_rate"]
    config["training"]["weight_decay"] = trial["weight_decay"]
    config["training"]["feedback_loss_weight"] = trial["feedback_loss_weight"]
    config["training"]["aware_loss_weight"] = trial["aware_loss_weight"]
    config["runtime"]["inference_mode"] = trial["inference_mode"]

    print(f"Running {dataset}/{trial['name']}/seed{seed}", flush=True)
    metrics = train_once(dataset, config, allow_download=False, log_every=0)
    row = {
        "dataset": dataset,
        "trial": trial["name"],
        "seed": seed,
        "accuracy": metrics["accuracy"],
        "f1_macro": metrics["f1_macro"],
        "auc": metrics["auc"],
        "best_val_accuracy": metrics["best_val_accuracy"],
        "best_epoch": metrics["best_epoch"],
        "elapsed_seconds": metrics["elapsed_seconds"],
        "inference_mode": metrics["inference_mode"],
        **trial,
    }
    all_rows.append(row)

    trial_payload = {"trial": trial, "seed": seed, "metrics": metrics}
    stem = f"{trial['name']}_seed{seed}"
    with (dataset_dir / f"{stem}.json").open("w", encoding="utf-8") as handle:
        json.dump(trial_payload, handle, indent=2, allow_nan=True)
    with (dataset_dir / f"{stem}.log").open("w", encoding="utf-8") as handle:
        handle.write(f"dataset: {dataset}\n")
        handle.write(f"trial: {trial['name']}\n")
        handle.write(f"seed: {seed}\n")
        handle.write(f"device_requested: {args.device}\n")
        handle.write(f"cuda_available: {torch.cuda.is_available()}\n")
        if torch.cuda.is_available():
            handle.write(f"cuda_device: {torch.cuda.get_device_name(0)}\n")
        handle.write(f"trial_config: {json.dumps(trial, sort_keys=True)}\n\n")
        for hist in metrics["history"]:
            handle.write(json.dumps(hist, allow_nan=True) + "\n")
        handle.write("\nfinal_metrics:\n")
        for key, value in metrics.items():
            if key != "history":
                handle.write(f"{key}: {value}\n")


def write_summary(
    output_dir: Path,
    all_rows: list[dict[str, Any]],
    best_rows: list[dict[str, Any]],
    args: argparse.Namespace,
    trials: list[dict[str, Any]],
) -> None:
    with (output_dir / "all_trials.json").open("w", encoding="utf-8") as handle:
        json.dump(all_rows, handle, indent=2, allow_nan=True)

    with (output_dir / "summary.md").open("w", encoding="utf-8") as handle:
        handle.write("# GNFBC Small Dataset Tuning Summary\n\n")
        handle.write(f"- Device requested: `{args.device}`\n")
        handle.write(f"- CUDA available: `{torch.cuda.is_available()}`\n")
        if torch.cuda.is_available():
            handle.write(f"- CUDA device: `{torch.cuda.get_device_name(0)}`\n")
        handle.write(f"- Epoch budget per trial: `{args.epochs}`\n")
        handle.write(f"- Early stopping patience: `{args.patience}`\n")
        handle.write(f"- Trial count per dataset: `{len(trials)}`\n")
        handle.write(f"- Seeds per trial: `{args.seeds}`\n\n")

        handle.write("## Best By Dataset\n\n")
        handle.write("| Dataset | Best Trial | Mode | Seed | Test Acc | Paper Acc | Gap | F1 Macro | AUC | Best Epoch | Seconds |\n")
        handle.write("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in best_rows:
            target = PAPER_TARGETS.get(row["dataset"], {}).get("accuracy")
            gap = None if target is None else row["accuracy"] - target
            handle.write(
                f"| {row['dataset']} | {row['trial']} | {row['inference_mode']} | {row['seed']} | {row['accuracy']:.4f} | "
                f"{format_float(target)} | {format_float(gap)} | {row['f1_macro']:.4f} | "
                f"{format_float(row['auc'])} | {row['best_epoch']:.0f} | {row['elapsed_seconds']:.2f} |\n"
            )

        handle.write("\n## Trial Grid\n\n")
        for trial in trials:
            handle.write(f"- `{trial['name']}`: `{json.dumps(trial, sort_keys=True)}`\n")

        handle.write("\n## Interpretation\n\n")
        handle.write(
            "- Paper targets come from `docs/GNFBC.pdf` Table 1 when available. "
            "Wisconsin and Washington are included as small WebKB sanity datasets, but the main paper table does not report them.\n"
        )
        handle.write(
            "- Each dataset folder contains one `.log` with full per-epoch JSON lines and one `.json` with the complete metrics payload per trial.\n"
        )
        handle.write(
            "- These sweeps tune core optimization settings only. If gaps remain large, the likely next step is adding the paper's additional backbone variants or reproducing its exact split/seed protocol.\n"
        )


def format_float(value: Any) -> str:
    if value is None:
        return "-"
    try:
        if math.isnan(float(value)):
            return "nan"
    except (TypeError, ValueError):
        return str(value)
    return f"{float(value):.4f}"


if __name__ == "__main__":
    main()
