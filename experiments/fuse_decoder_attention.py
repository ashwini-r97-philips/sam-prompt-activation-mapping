from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def normalize(x):
    x = x.astype(np.float32)
    x = x - x.min()
    return x / (x.max() + 1e-8)


def save_map(arr, path, title):
    plt.figure(figsize=(6, 6))
    plt.imshow(normalize(arr), cmap="viridis", vmin=0, vmax=1)
    plt.axis("off")
    plt.title(title)
    plt.colorbar()
    plt.savefig(path, bbox_inches="tight", dpi=200)
    plt.close()


def parse_layer(path: Path):
    m = re.search(r"decoder_layer_(\d+)_query(\d+)_attn\.npy", path.name)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--decoder-attn-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--query-index", type=int, default=None)
    parser.add_argument("--weights-csv", default=None)
    args = parser.parse_args()

    attn_dir = Path(args.decoder_attn_dir)
    out_dir = Path(args.out_dir) / "fused_decoder_attention"
    out_dir.mkdir(parents=True, exist_ok=True)

    maps = []
    rows = []

    for p in sorted(attn_dir.glob("decoder_layer_*_query*_attn.npy")):
        parsed = parse_layer(p)
        if parsed is None:
            continue
        layer, query = parsed
        if args.query_index is not None and query != args.query_index:
            continue

        arr = np.load(p).astype(np.float32)
        maps.append((layer, query, arr, p))

    if not maps:
        raise RuntimeError(f"No decoder attention maps found in {attn_dir}")

    maps = sorted(maps, key=lambda x: x[0])
    query = maps[0][1]

    stack = np.stack([m[2] for m in maps], axis=0)
    fused_mean = stack.mean(axis=0)

    mean_npy = out_dir / f"decoder_query{query}_fused_mean_attn.npy"
    mean_png = out_dir / f"decoder_query{query}_fused_mean_attn.png"

    np.save(mean_npy, fused_mean)
    save_map(fused_mean, mean_png, f"Query {query} Fused Mean Decoder Attention")

    rows.append(
        {
            "query_index": query,
            "fusion_type": "mean",
            "num_layers": len(maps),
            "mean": float(fused_mean.mean()),
            "max": float(fused_mean.max()),
            "npy_path": str(mean_npy),
            "png_path": str(mean_png),
        }
    )

    if args.weights_csv is not None:
        layer_weights = {}
        with open(args.weights_csv, newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                if "layer" in r and "top_score_drop" in r:
                    layer_weights[int(r["layer"])] = max(float(r["top_score_drop"]), 0.0)

        weights = np.array([layer_weights.get(m[0], 0.0) for m in maps], dtype=np.float32)
        if weights.sum() <= 0:
            weights = np.ones(len(maps), dtype=np.float32)
        weights = weights / weights.sum()

        fused_weighted = (stack * weights[:, None, None]).sum(axis=0)

        weighted_npy = out_dir / f"decoder_query{query}_fused_weighted_attn.npy"
        weighted_png = out_dir / f"decoder_query{query}_fused_weighted_attn.png"

        np.save(weighted_npy, fused_weighted)
        save_map(
            fused_weighted,
            weighted_png,
            f"Query {query} Weighted Fused Decoder Attention",
        )

        rows.append(
            {
                "query_index": query,
                "fusion_type": "weighted_by_attention_ablation",
                "num_layers": len(maps),
                "mean": float(fused_weighted.mean()),
                "max": float(fused_weighted.max()),
                "npy_path": str(weighted_npy),
                "png_path": str(weighted_png),
            }
        )

    csv_path = out_dir / "fused_decoder_attention_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with open(out_dir / "run_config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"Saved: {csv_path}")
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()