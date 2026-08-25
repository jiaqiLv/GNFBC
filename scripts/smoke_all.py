from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gnfbc.datasets import DATASETS
from gnfbc.train import train_once
from gnfbc.utils import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a short GNFBC training pass on datasets.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--hidden-channels", type=int, default=16)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--datasets", nargs="*", default=DATASETS)
    args = parser.parse_args()

    config = load_yaml(args.config)
    config["training"]["epochs"] = args.epochs
    config["training"]["patience"] = max(args.epochs, 1)
    config["runtime"]["device"] = args.device
    config["model"]["hidden_channels"] = args.hidden_channels
    if args.data_root is not None:
        config["data"]["root"] = args.data_root

    for dataset in args.datasets:
        try:
            metrics = train_once(dataset, config, allow_download=False)
            print(
                f"{dataset}: ok "
                f"acc={metrics['accuracy']:.4f} "
                f"f1={metrics['f1_macro']:.4f} "
                f"auc={metrics['auc']:.4f}"
            )
        except Exception as exc:
            print(f"{dataset}: failed: {exc}")


if __name__ == "__main__":
    main()
