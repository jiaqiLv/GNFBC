# GNFBC

Reproduction code for **Graph Negative Feedback Bias Correction Framework for Adaptive Heterophily Modeling**.

The implementation focuses on the paper's main setup:

- GraphSAGE is used as the graph-aware backbone.
- A graph-agnostic counterpart shares the same linear weights.
- Per-node feedback coefficients are computed from Dirichlet energy.
- Training uses graph-aware/agnostic correction and the negative feedback loss.
- Inference uses only the graph-aware branch, matching the paper's no-extra-inference-cost design.

## Project Layout

```text
configs/              Default experiment configuration
data/                 Downloaded raw files and processed graph caches
deprecated/MANDATE/   Previous related project, kept as reference only
docs/GNFBC.pdf        Source paper
gnfbc/                New GNFBC implementation
scripts/              Dataset download and validation helpers
tools/                Local utility scripts
```

## Environment

Create and install the project environment in the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -U pip
.\.venv\Scripts\python -m pip install -r requirements.txt
```

## Download Datasets

```powershell
.\.venv\Scripts\python scripts\download_data.py --all
```

The script downloads and normalizes the datasets used by the paper:

`cora`, `citeseer`, `pubmed`, `computers`, `photo`, `chameleon`, `squirrel`, `wisconsin`, `washington`, `texas`, `cornell`, `yelpchi`, and `amazon_fraud`.

## Train

```powershell
.\.venv\Scripts\python -m gnfbc.train --dataset texas --epochs 200 --device cpu
```

Use `--config configs/default.yaml` to start from the default paper-style settings and override values from the command line.

## Smoke Test

Run a short pass on every available dataset:

```powershell
.\.venv\Scripts\python scripts\smoke_all.py --epochs 1 --device cpu
```
