from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gnfbc.datasets import DATASETS, download_dataset, normalize_dataset_name


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and cache GNFBC datasets.")
    parser.add_argument("datasets", nargs="*", help="Dataset names to download.")
    parser.add_argument("--all", action="store_true", help="Download all paper datasets.")
    parser.add_argument("--root", default="data", help="Dataset root directory.")
    parser.add_argument("--force", action="store_true", help="Re-download raw files.")
    args = parser.parse_args()

    names = DATASETS if args.all else [normalize_dataset_name(name) for name in args.datasets]
    if not names:
        raise SystemExit("Specify dataset names or pass --all.")

    for name in names:
        path = download_dataset(name, root=args.root, force=args.force)
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
