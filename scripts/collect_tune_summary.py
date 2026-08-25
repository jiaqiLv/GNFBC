from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


PAPER_TARGETS = {
    "cora": 0.8656,
    "citeseer": 0.7391,
    "pubmed": 0.8930,
    "computers": 0.8043,
    "photo": 0.8954,
    "chameleon": 0.6742,
    "squirrel": 0.5931,
    "texas": 0.8990,
    "cornell": 0.8889,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect a tune run summary from trial JSON files.")
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()

    rows = []
    for path in sorted(args.run_dir.glob("*/*.json")):
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        trial = payload["trial"]
        metrics = payload["metrics"]
        rows.append(
            {
                "dataset": path.parent.name,
                "trial": trial["name"],
                "seed": payload["seed"],
                "mode": metrics.get("inference_mode", trial.get("inference_mode", "aware")),
                "accuracy": metrics["accuracy"],
                "f1_macro": metrics["f1_macro"],
                "auc": metrics["auc"],
                "best_val_accuracy": metrics["best_val_accuracy"],
                "best_epoch": metrics["best_epoch"],
                "elapsed_seconds": metrics["elapsed_seconds"],
            }
        )

    best_rows = []
    for dataset in sorted({row["dataset"] for row in rows}):
        candidates = [row for row in rows if row["dataset"] == dataset]
        best_rows.append(max(candidates, key=lambda row: row["accuracy"]))

    with (args.run_dir / "all_trials_collected.json").open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2, allow_nan=True)

    with (args.run_dir / "summary.md").open("w", encoding="utf-8") as handle:
        handle.write("# GNFBC Collected Tuning Summary\n\n")
        handle.write(f"- Run directory: `{args.run_dir}`\n")
        handle.write(f"- Completed trial files: `{len(rows)}`\n\n")
        handle.write("| Dataset | Best Trial | Mode | Seed | Test Acc | Paper Acc | Gap | F1 Macro | AUC | Best Epoch | Seconds |\n")
        handle.write("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in best_rows:
            target = PAPER_TARGETS.get(row["dataset"])
            gap = None if target is None else row["accuracy"] - target
            handle.write(
                f"| {row['dataset']} | {row['trial']} | {row['mode']} | {row['seed']} | "
                f"{row['accuracy']:.4f} | {fmt(target)} | {fmt(gap)} | "
                f"{row['f1_macro']:.4f} | {fmt(row['auc'])} | "
                f"{row['best_epoch']:.0f} | {row['elapsed_seconds']:.2f} |\n"
            )
        handle.write("\n## Notes\n\n")
        handle.write("- This summary was rebuilt from completed per-trial JSON files.\n")
        handle.write("- It may represent a partial run if the original sweep was interrupted.\n")

    print(args.run_dir / "summary.md")


def fmt(value: Any) -> str:
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
