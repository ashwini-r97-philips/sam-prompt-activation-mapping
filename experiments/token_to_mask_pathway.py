from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def normalize(arr):
    arr = arr.astype(np.float32)
    arr = arr - arr.min()
    return arr / (arr.max() + 1e-8)


def cosine(a, b):
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12))


def pearson(a, b):
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    a = a - a.mean()
    b = b - b.mean()
    return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12))


def save_map(arr, path, title):
    plt.figure(figsize=(6, 6))
    plt.imshow(normalize(arr), cmap="viridis", vmin=0, vmax=1)
    plt.axis("off")
    plt.title(title)
    plt.colorbar()
    plt.savefig(path, bbox_inches="tight", dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--token-fusion-dir", required=True)
    parser.add_argument("--mask-region-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--target-desc", default="mean_mask_logit")
    parser.add_argument("--max-prompt-tokens", type=int, default=16)
    args = parser.parse_args()

    token_fusion_dir = Path(args.token_fusion_dir)
    mask_region_dir = Path(args.mask_region_dir)
    out_dir = Path(args.out_dir) / "token_to_mask_pathway"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for layer in range(6):
        region_path = mask_region_dir / f"layer_{layer}_{args.target_desc}_fusion_grad.npy"
        if not region_path.is_file():
            raise FileNotFoundError(f"Missing region grad map: {region_path}")

        region_grad = np.load(region_path)

        for token_idx in range(args.max_prompt_tokens):
            attn_path = token_fusion_dir / f"layer_{layer}_prompt_token_{token_idx}_attn.npy"
            if not attn_path.is_file():
                continue

            attn = np.load(attn_path)

            if attn.shape != region_grad.shape:
                raise RuntimeError(f"Shape mismatch: {attn.shape} vs {region_grad.shape}")

            pathway = attn * region_grad

            npy_path = out_dir / f"layer_{layer}_prompt_token_{token_idx}_to_{args.target_desc}.npy"
            png_path = out_dir / f"layer_{layer}_prompt_token_{token_idx}_to_{args.target_desc}.png"

            np.save(npy_path, pathway)
            save_map(
                pathway,
                png_path,
                f"Layer {layer} Prompt Token {token_idx} → {args.target_desc}",
            )

            flat = pathway.reshape(-1)
            top_idx = int(np.argmax(flat))
            row = top_idx // pathway.shape[1]
            col = top_idx % pathway.shape[1]

            rows.append(
                {
                    "layer": layer,
                    "prompt_token_index": token_idx,
                    "target_desc": args.target_desc,
                    "pathway_mean": float(pathway.mean()),
                    "pathway_max": float(pathway.max()),
                    "top_token_index": top_idx,
                    "top_row": row,
                    "top_col": col,
                    "cosine_attn_regiongrad": cosine(attn, region_grad),
                    "pearson_attn_regiongrad": pearson(attn, region_grad),
                    "npy_path": str(npy_path),
                    "png_path": str(png_path),
                }
            )

    rows_sorted = sorted(rows, key=lambda r: r["pathway_max"], reverse=True)

    csv_path = out_dir / "token_to_mask_pathway.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_sorted[0].keys()))
        writer.writeheader()
        writer.writerows(rows_sorted)

    json_path = out_dir / "top_token_to_mask_pathways.json"
    with open(json_path, "w") as f:
        json.dump(rows_sorted[:30], f, indent=2)

    print(f"Saved token-to-mask pathway CSV: {csv_path}")
    print(f"Saved top pathways JSON: {json_path}")
    print()
    print("Top token-to-mask pathways:")
    for r in rows_sorted[:10]:
        print(
            f"Layer {r['layer']} token {r['prompt_token_index']} | "
            f"max={r['pathway_max']:.6e} | "
            f"top=({r['top_row']},{r['top_col']})"
        )


if __name__ == "__main__":
    main()