from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def normalize_flat(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64).reshape(-1)
    x = x - x.min()
    denom = x.max() + 1e-12
    return x / denom


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)

    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denom)


def pearson_correlation(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)

    a = a - a.mean()
    b = b - b.mean()

    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denom)


def topk_indices(x: np.ndarray, k: int) -> set[int]:
    flat = x.reshape(-1)
    k = min(k, flat.size)
    idx = np.argsort(flat)[::-1][:k]
    return set(int(i) for i in idx)


def topk_overlap(a: np.ndarray, b: np.ndarray, k: int) -> float:
    a_top = topk_indices(a, k)
    b_top = topk_indices(b, k)

    if not a_top or not b_top:
        return 0.0

    return len(a_top.intersection(b_top)) / float(k)


def load_map(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Missing file: {path}")

    arr = np.load(path)

    if arr.ndim != 2:
        raise RuntimeError(f"Expected 2D map, got {arr.shape} from {path}")

    return arr.astype(np.float32)


def infer_query_index(attn_dir: Path) -> int:
    mapping_path = attn_dir / "query_mapping.json"

    if not mapping_path.is_file():
        raise FileNotFoundError(f"Missing query mapping: {mapping_path}")

    with open(mapping_path, "r") as f:
        mapping = json.load(f)

    if "used_query_index" in mapping:
        return int(mapping["used_query_index"])

    if "responsible_query" in mapping:
        return int(mapping["responsible_query"])

    raise RuntimeError(f"Could not find query index in {mapping_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--delta-grad-dir",
        required=True,
        help="Directory containing layer_i_delta_grad.npy files.",
    )
    parser.add_argument(
        "--decoder-attn-dir",
        required=True,
        help="Directory containing decoder_layer_i_query*_attn.npy files.",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Output directory for pathway consistency results.",
    )
    parser.add_argument(
        "--query-index",
        type=int,
        default=None,
        help="Optional query index. If omitted, read from query_mapping.json.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        nargs="+",
        default=[20, 50, 100, 200],
        help="Top-k values for overlap calculation.",
    )
    args = parser.parse_args()

    delta_grad_dir = Path(args.delta_grad_dir)
    decoder_attn_dir = Path(args.decoder_attn_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    query_index = args.query_index
    if query_index is None:
        query_index = infer_query_index(decoder_attn_dir)

    rows = []

    for layer in range(6):
        delta_grad_path = delta_grad_dir / f"layer_{layer}_delta_grad.npy"
        decoder_attn_path = (
            decoder_attn_dir / f"decoder_layer_{layer}_query{query_index}_attn.npy"
        )

        delta_grad = load_map(delta_grad_path)
        decoder_attn = load_map(decoder_attn_path)

        if delta_grad.shape != decoder_attn.shape:
            raise RuntimeError(
                f"Shape mismatch at layer {layer}: "
                f"delta_grad={delta_grad.shape}, decoder_attn={decoder_attn.shape}"
            )

        delta_norm = normalize_flat(delta_grad)
        attn_norm = normalize_flat(decoder_attn)

        row = {
            "layer": layer,
            "query_index": query_index,
            "spatial_shape": str(delta_grad.shape),
            "delta_grad_path": str(delta_grad_path),
            "decoder_attn_path": str(decoder_attn_path),
            "cosine_raw": cosine_similarity(delta_grad, decoder_attn),
            "cosine_normalized": cosine_similarity(delta_norm, attn_norm),
            "pearson_raw": pearson_correlation(delta_grad, decoder_attn),
            "pearson_normalized": pearson_correlation(delta_norm, attn_norm),
            "delta_grad_mean": float(delta_grad.mean()),
            "delta_grad_max": float(delta_grad.max()),
            "decoder_attn_mean": float(decoder_attn.mean()),
            "decoder_attn_max": float(decoder_attn.max()),
        }

        for k in args.top_k:
            row[f"top_{k}_overlap"] = topk_overlap(delta_grad, decoder_attn, k)

        rows.append(row)

    csv_path = out_dir / "pathway_consistency.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "query_index": query_index,
        "delta_grad_dir": str(delta_grad_dir),
        "decoder_attn_dir": str(decoder_attn_dir),
        "out_dir": str(out_dir),
        "num_layers": 6,
        "top_k": args.top_k,
        "mean_cosine_normalized": float(
            np.mean([r["cosine_normalized"] for r in rows])
        ),
        "mean_pearson_normalized": float(
            np.mean([r["pearson_normalized"] for r in rows])
        ),
    }

    json_path = out_dir / "pathway_consistency_summary.json"

    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved pathway consistency CSV: {csv_path}")
    print(f"Saved pathway consistency summary: {json_path}")
    print()
    print("Layer results:")
    for row in rows:
        print(
            f"Layer {row['layer']} | "
            f"cos_norm={row['cosine_normalized']:.4f} | "
            f"corr_norm={row['pearson_normalized']:.4f} | "
            f"top20={row.get('top_20_overlap', 0.0):.3f} | "
            f"top100={row.get('top_100_overlap', 0.0):.3f}"
        )


if __name__ == "__main__":
    main()