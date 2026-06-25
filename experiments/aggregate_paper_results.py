from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pandas as pd


def read_csv_safe(path):
    path = Path(path)
    if not path.is_file():
        return None
    return pd.read_csv(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    examples = list(csv.DictReader(open(args.manifest)))

    summary_rows = []

    for ex in examples:
        example_id = ex["example_id"]
        class_name = ex["class_name"]
        base = Path(args.out_root) / example_id

        row = {
            "example_id": example_id,
            "class_name": class_name,
            "image": ex["image"],
            "prompt": ex["prompt"],
        }

        qmap = base / "decoder_query_trace" / "decoder_query_trace" / "query_mapping.json"
        if qmap.is_file():
            data = json.load(open(qmap))
            row["responsible_query"] = data.get("responsible_query", data.get("used_query_index"))

        qcsv = read_csv_safe(base / "decoder_query_trace" / "decoder_query_trace" / "query_gradient_scores.csv")
        if qcsv is not None:
            nonzero = (qcsv["l2_grad"] > 0).sum()
            top = qcsv.sort_values("l2_grad", ascending=False).iloc[0]
            row["top_query"] = int(top["query_index"])
            row["top_query_l2_grad"] = float(top["l2_grad"])
            row["num_nonzero_queries"] = int(nonzero)

        dfg = read_csv_safe(base / "multimodal_deltaf_grad" / "mm_grad" / "delta_grad_scores.csv")
        if dfg is not None:
            best = dfg.sort_values("mean_delta_grad", ascending=False).iloc[0]
            row["best_delta_grad_layer"] = int(best["layer"])
            row["best_delta_grad_mean"] = float(best["mean_delta_grad"])

        pc = read_csv_safe(base / "pathway_consistency" / "pathway_consistency.csv")
        if pc is not None:
            best = pc.sort_values("cosine_normalized", ascending=False).iloc[0]
            row["best_pathway_consistency_layer"] = int(best["layer"])
            row["best_pathway_cosine"] = float(best["cosine_normalized"])
            row["best_pathway_corr"] = float(best["pearson_normalized"])

        pti = read_csv_safe(base / "prompt_token_importance" / "prompt_token_importance" / "prompt_token_importance.csv")
        if pti is not None:
            top = pti.sort_values("l2_grad", ascending=False).iloc[0]
            row["top_prompt_token"] = int(top["prompt_token_index"])
            row["top_prompt_token_norm"] = float(top["normalized_l2_grad"])

        summary_rows.append(row)

    df = pd.DataFrame(summary_rows)
    df.to_csv(out_dir / "paper_example_summary.csv", index=False)

    print(f"Saved: {out_dir / 'paper_example_summary.csv'}")
    print(df)


if __name__ == "__main__":
    main()