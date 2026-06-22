from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_query_scores(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing query score CSV: {path}")

    df = pd.read_csv(path)

    required = {
        "query_index",
        "l2_grad",
        "mean_abs_grad",
        "max_abs_grad",
        "normalized_l2_grad",
        "is_responsible_query",
    }

    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing columns from {path}: {sorted(missing)}")

    return df


def save_top_query_plot(df: pd.DataFrame, out_path: Path, title: str, top_n: int):
    top = df.sort_values("l2_grad", ascending=False).head(top_n)

    labels = [str(int(q)) for q in top["query_index"].tolist()]
    values = top["normalized_l2_grad"].tolist()

    plt.figure(figsize=(8, 4))
    plt.bar(labels, values)
    plt.xlabel("Query index")
    plt.ylabel("Normalized L2 gradient")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def summarize_one(name: str, csv_path: Path, out_dir: Path, top_n: int):
    df = load_query_scores(csv_path)

    df_sorted = df.sort_values("l2_grad", ascending=False).reset_index(drop=True)

    top_rows = []
    for rank, row in df_sorted.head(top_n).iterrows():
        top_rows.append(
            {
                "example": name,
                "rank": int(rank + 1),
                "query_index": int(row["query_index"]),
                "l2_grad": float(row["l2_grad"]),
                "mean_abs_grad": float(row["mean_abs_grad"]),
                "max_abs_grad": float(row["max_abs_grad"]),
                "normalized_l2_grad": float(row["normalized_l2_grad"]),
                "is_responsible_query": int(row["is_responsible_query"]),
            }
        )

    q1 = df_sorted.iloc[0]
    q2 = df_sorted.iloc[1] if len(df_sorted) > 1 else None

    top1_l2 = float(q1["l2_grad"])
    top2_l2 = float(q2["l2_grad"]) if q2 is not None else 0.0

    dominance_ratio = top1_l2 / (top2_l2 + 1e-12)

    num_nonzero = int((df["l2_grad"] > 0).sum())

    summary = {
        "example": name,
        "top_query": int(q1["query_index"]),
        "top_query_l2_grad": top1_l2,
        "second_query": int(q2["query_index"]) if q2 is not None else None,
        "second_query_l2_grad": top2_l2,
        "dominance_ratio_top1_vs_top2": float(dominance_ratio),
        "num_queries": int(len(df)),
        "num_nonzero_queries": num_nonzero,
        "csv_path": str(csv_path),
    }

    plot_path = out_dir / f"{name}_top_queries.png"
    save_top_query_plot(
        df=df,
        out_path=plot_path,
        title=f"{name}: top decoder query gradients",
        top_n=top_n,
    )
    summary["plot_path"] = str(plot_path)

    return summary, top_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--schoolbus-csv", required=True)
    parser.add_argument("--cat-csv", required=True)
    parser.add_argument("--dog-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--top-n", type=int, default=20)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    inputs = [
        ("schoolbus", Path(args.schoolbus_csv)),
        ("cat", Path(args.cat_csv)),
        ("dog", Path(args.dog_csv)),
    ]

    summaries = []
    all_top_rows = []

    for name, csv_path in inputs:
        summary, top_rows = summarize_one(
            name=name,
            csv_path=csv_path,
            out_dir=out_dir,
            top_n=args.top_n,
        )
        summaries.append(summary)
        all_top_rows.extend(top_rows)

    summary_csv = out_dir / "query_144_summary.csv"
    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)

    top_csv = out_dir / "top_query_gradients.csv"
    with open(top_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_top_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_top_rows)

    json_path = out_dir / "query_144_summary.json"
    with open(json_path, "w") as f:
        json.dump(summaries, f, indent=2)

    print(f"Saved query 144 summary CSV: {summary_csv}")
    print(f"Saved top query gradients CSV: {top_csv}")
    print(f"Saved query 144 summary JSON: {json_path}")
    print()
    print("Summary:")
    for row in summaries:
        print(
            f"{row['example']}: "
            f"top query={row['top_query']} | "
            f"top l2={row['top_query_l2_grad']:.6e} | "
            f"second query={row['second_query']} | "
            f"second l2={row['second_query_l2_grad']:.6e} | "
            f"dominance={row['dominance_ratio_top1_vs_top2']:.2e} | "
            f"nonzero queries={row['num_nonzero_queries']}"
        )


if __name__ == "__main__":
    main()