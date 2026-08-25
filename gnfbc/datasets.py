from __future__ import annotations

import pickle
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import scipy.io as sio
import scipy.sparse as sp
import torch

from gnfbc.graph import GraphData
from gnfbc.utils import (
    edge_index_from_sparse,
    row_normalize,
    scipy_to_torch_dense,
    stratified_masks,
)


PLANETOID_URL = "https://github.com/kimiyoung/planetoid/raw/master/data"
GEOM_GCN_URL = (
    "https://raw.githubusercontent.com/graphdml-uiuc-jlu/geom-gcn/"
    "f1fc0d14b3b019c562737240d06ec83b07d16a8f/new_data"
)
GNN_BENCHMARK_URL = "https://github.com/shchur/gnn-benchmark/raw/master/data/npz"
DGL_DATA_URL = "https://data.dgl.ai/dataset"
WEBKB_LINQS_URL = "https://linqs-data.soe.ucsc.edu/public/lbc/WebKB.tgz"

PLANETOID = {"cora", "citeseer", "pubmed"}
WEBKB = {"cornell", "texas", "wisconsin", "washington"}
WIKI = {"chameleon", "squirrel"}
AMAZON_PRODUCT = {
    "computers": "amazon_electronics_computers.npz",
    "photo": "amazon_electronics_photo.npz",
}
FRAUD = {
    "yelpchi": ("FraudYelp.zip", "YelpChi.mat"),
    "amazon_fraud": ("FraudAmazon.zip", "Amazon.mat"),
}

DATASETS = sorted(PLANETOID | WEBKB | WIKI | set(AMAZON_PRODUCT) | set(FRAUD))


def download_dataset(name: str, root: str | Path = "data", force: bool = False) -> Path:
    name = normalize_dataset_name(name)
    root = Path(root)
    processed_path = root / "processed" / f"{name}.pt"
    if processed_path.exists() and not force:
        return processed_path

    if name in PLANETOID:
        _download_planetoid(name, root, force)
        graph = _load_planetoid(name, root)
    elif name == "washington":
        _download_webkb_linqs(root, force)
        graph = _load_webkb_linqs(name, root)
    elif name in WEBKB or name in WIKI:
        _download_geom_gcn(name, root, force)
        graph = _load_geom_gcn(name, root)
    elif name in AMAZON_PRODUCT:
        _download_amazon_product(name, root, force)
        graph = _load_amazon_product(name, root)
    elif name in FRAUD:
        _download_fraud(name, root, force)
        graph = _load_fraud(name, root)
    else:
        raise ValueError(f"Unknown dataset: {name}")

    processed_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(graph, processed_path)
    return processed_path


def load_dataset(
    name: str,
    root: str | Path = "data",
    train_ratio: float = 0.4,
    val_ratio: float = 0.2,
    seed: int = 2,
    download: bool = True,
) -> GraphData:
    name = normalize_dataset_name(name)
    processed_path = Path(root) / "processed" / f"{name}.pt"
    if not processed_path.exists():
        if not download:
            raise FileNotFoundError(f"Missing processed dataset: {processed_path}")
        download_dataset(name, root)

    payload = torch.load(processed_path, weights_only=False)
    x = payload["x"].float()
    edge_index = payload["edge_index"].long()
    y = payload["y"].long()
    train_mask, val_mask, test_mask = stratified_masks(y, train_ratio, val_ratio, seed)
    return GraphData(
        name=name,
        x=x,
        edge_index=edge_index,
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        num_classes=int(y.max().item() + 1),
    )


def normalize_dataset_name(name: str) -> str:
    clean = name.strip().lower().replace("-", "_")
    aliases = {
        "citeseer": "citeseer",
        "cite_seer": "citeseer",
        "amazon": "amazon_fraud",
        "fraudamazon": "amazon_fraud",
        "fraud_amazon": "amazon_fraud",
        "yelphi": "yelpchi",
        "fraudyelp": "yelpchi",
        "fraud_yelp": "yelpchi",
    }
    return aliases.get(clean, clean)


def _download(url: str, path: Path, force: bool = False) -> None:
    if path.exists() and not force:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url}", flush=True)
    urllib.request.urlretrieve(url, path)


def _download_planetoid(name: str, root: Path, force: bool) -> None:
    folder = root / "raw" / "planetoid"
    for suffix in ["x", "tx", "allx", "y", "ty", "ally", "graph", "test.index"]:
        _download(f"{PLANETOID_URL}/ind.{name}.{suffix}", folder / f"ind.{name}.{suffix}", force)


def _load_planetoid(name: str, root: Path) -> dict[str, torch.Tensor]:
    folder = root / "raw" / "planetoid"
    objects = []
    for suffix in ["x", "tx", "allx", "y", "ty", "ally", "graph"]:
        with (folder / f"ind.{name}.{suffix}").open("rb") as handle:
            objects.append(pickle.load(handle, encoding="latin1"))
    x, tx, allx, y, ty, ally, graph = objects
    test_idx = np.loadtxt(folder / f"ind.{name}.test.index", dtype=np.int64)
    test_idx_sorted = np.sort(test_idx)

    if name == "citeseer":
        full_range = range(test_idx.min(), test_idx.max() + 1)
        tx_ext = sp.lil_matrix((len(full_range), x.shape[1]))
        tx_ext[test_idx_sorted - test_idx.min(), :] = tx
        ty_ext = np.zeros((len(full_range), y.shape[1]))
        ty_ext[test_idx_sorted - test_idx.min(), :] = ty
        tx, ty = tx_ext, ty_ext

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx, :] = features[test_idx_sorted, :]
    labels = np.vstack((ally, ty))
    labels[test_idx, :] = labels[test_idx_sorted, :]
    adj = _graph_dict_to_sparse(graph)
    return _make_payload(features, adj, labels.argmax(axis=1))


def _graph_dict_to_sparse(graph: dict[int, list[int]]) -> sp.csr_matrix:
    rows: list[int] = []
    cols: list[int] = []
    for src, neighbors in graph.items():
        rows.extend([src] * len(neighbors))
        cols.extend(neighbors)
    size = max(max(rows, default=0), max(cols, default=0)) + 1
    data = np.ones(len(rows), dtype=np.float32)
    return sp.coo_matrix((data, (rows, cols)), shape=(size, size)).tocsr()


def _download_geom_gcn(name: str, root: Path, force: bool) -> None:
    folder = root / "raw" / "geom_gcn" / name
    for filename in ["out1_node_feature_label.txt", "out1_graph_edges.txt"]:
        _download(f"{GEOM_GCN_URL}/{name}/{filename}", folder / filename, force)


def _load_geom_gcn(name: str, root: Path) -> dict[str, torch.Tensor]:
    folder = root / "raw" / "geom_gcn" / name
    node_file = folder / "out1_node_feature_label.txt"
    edge_file = folder / "out1_graph_edges.txt"
    features: dict[int, np.ndarray] = {}
    labels: dict[int, int] = {}
    with node_file.open("r", encoding="utf-8") as handle:
        next(handle)
        for line in handle:
            node_id, feature_text, label_text = line.rstrip().split("\t")
            features[int(node_id)] = np.fromstring(feature_text, dtype=np.float32, sep=",")
            labels[int(node_id)] = int(label_text)

    rows: list[int] = []
    cols: list[int] = []
    with edge_file.open("r", encoding="utf-8") as handle:
        next(handle)
        for line in handle:
            src, dst = line.rstrip().split("\t")
            rows.append(int(src))
            cols.append(int(dst))

    node_ids = sorted(features)
    remap = {node_id: i for i, node_id in enumerate(node_ids)}
    x = np.vstack([features[node_id] for node_id in node_ids])
    y = np.asarray([labels[node_id] for node_id in node_ids], dtype=np.int64)
    row: list[int] = []
    col: list[int] = []
    for src, dst in zip(rows, cols):
        if src in remap and dst in remap:
            row.append(remap[src])
            col.append(remap[dst])
    adj = sp.coo_matrix((np.ones(len(row)), (row, col)), shape=(len(node_ids), len(node_ids)))
    return _make_payload(x, adj, y)


def _download_webkb_linqs(root: Path, force: bool) -> None:
    path = root / "raw" / "webkb_linqs" / "WebKB.tgz"
    _download(WEBKB_LINQS_URL, path, force)


def _load_webkb_linqs(name: str, root: Path) -> dict[str, torch.Tensor]:
    path = root / "raw" / "webkb_linqs" / "WebKB.tgz"
    label_map = {
        "course": 0,
        "faculty": 1,
        "student": 2,
        "project": 3,
        "staff": 4,
    }
    features: dict[str, np.ndarray] = {}
    labels: dict[str, int] = {}
    with tarfile.open(path, "r:gz") as archive:
        content = archive.extractfile(f"webkb/{name}.content")
        if content is None:
            raise FileNotFoundError(f"webkb/{name}.content not found in {path}")
        for raw_line in content:
            parts = raw_line.decode("utf-8").strip().split()
            if not parts:
                continue
            node_id = parts[0]
            features[node_id] = np.asarray(parts[1:-1], dtype=np.float32)
            labels[node_id] = label_map[parts[-1]]

        cites = archive.extractfile(f"webkb/{name}.cites")
        if cites is None:
            raise FileNotFoundError(f"webkb/{name}.cites not found in {path}")
        edges = [raw_line.decode("utf-8").strip().split() for raw_line in cites]

    node_ids = sorted(features)
    remap = {node_id: idx for idx, node_id in enumerate(node_ids)}
    row: list[int] = []
    col: list[int] = []
    for src, dst in edges:
        if src in remap and dst in remap:
            row.append(remap[src])
            col.append(remap[dst])
    x = np.vstack([features[node_id] for node_id in node_ids])
    y = np.asarray([labels[node_id] for node_id in node_ids], dtype=np.int64)
    adj = sp.coo_matrix((np.ones(len(row)), (row, col)), shape=(len(node_ids), len(node_ids)))
    return _make_payload(x, adj, y)


def _download_amazon_product(name: str, root: Path, force: bool) -> None:
    filename = AMAZON_PRODUCT[name]
    _download(f"{GNN_BENCHMARK_URL}/{filename}", root / "raw" / "amazon_product" / filename, force)


def _load_amazon_product(name: str, root: Path) -> dict[str, torch.Tensor]:
    path = root / "raw" / "amazon_product" / AMAZON_PRODUCT[name]
    loader = np.load(path, allow_pickle=True)
    adj = sp.csr_matrix(
        (loader["adj_data"], loader["adj_indices"], loader["adj_indptr"]),
        shape=loader["adj_shape"],
    )
    attr = sp.csr_matrix(
        (loader["attr_data"], loader["attr_indices"], loader["attr_indptr"]),
        shape=loader["attr_shape"],
    )
    labels = loader["labels"]
    return _make_payload(attr, adj, labels)


def _download_fraud(name: str, root: Path, force: bool) -> None:
    archive, _ = FRAUD[name]
    zip_path = root / "raw" / "fraud" / archive
    _download(f"{DGL_DATA_URL}/{archive}", zip_path, force)
    extract_dir = root / "raw" / "fraud" / name
    if force or not extract_dir.exists():
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as zipped:
            zipped.extractall(extract_dir)


def _load_fraud(name: str, root: Path) -> dict[str, torch.Tensor]:
    _, mat_name = FRAUD[name]
    candidates = list((root / "raw" / "fraud" / name).rglob(mat_name))
    if not candidates:
        raise FileNotFoundError(f"Could not find {mat_name} after extracting {name}")
    mat = sio.loadmat(candidates[0])
    features = mat["features"]
    labels = np.asarray(mat["label"]).reshape(-1)
    if "homo" in mat:
        adj = mat["homo"]
    else:
        relation_keys = [key for key in mat if key.startswith("net_")]
        if not relation_keys:
            raise KeyError(f"No adjacency matrix found in {candidates[0]}")
        adj = sum(mat[key] for key in relation_keys)
    return _make_payload(features, adj, labels)


def _make_payload(
    features: sp.spmatrix | np.ndarray,
    adj: sp.spmatrix,
    labels: np.ndarray,
) -> dict[str, torch.Tensor]:
    features = row_normalize(features)
    y = torch.tensor(np.asarray(labels), dtype=torch.long).contiguous()
    return {
        "x": scipy_to_torch_dense(features),
        "edge_index": edge_index_from_sparse(adj),
        "y": y,
    }


def print_dataset_list() -> None:
    for name in DATASETS:
        print(name)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print_dataset_list()
    else:
        download_dataset(sys.argv[1])
