from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch


def main() -> None:
    root = Path("data/processed")
    for path in sorted(root.glob("*.pt")):
        data = torch.load(path, weights_only=False)
        x = data["x"]
        edge_index = data["edge_index"]
        y = data["y"]
        print(
            f"{path.stem}: "
            f"nodes={x.shape[0]} "
            f"features={x.shape[1]} "
            f"edges={edge_index.shape[1]} "
            f"classes={int(y.max().item() + 1)}"
        )


if __name__ == "__main__":
    main()
