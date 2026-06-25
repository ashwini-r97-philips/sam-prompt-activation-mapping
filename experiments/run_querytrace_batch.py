from __future__ import annotations

import argparse
import csv
import os
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cuda-visible-devices", default=None)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    env = os.environ.copy()
    if args.cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    rows = list(csv.DictReader(open(args.manifest)))
    failures = []

    for row in rows:
        example_id = row["example_id"]
        image = row["image"]
        prompt = row["prompt"]

        out_dir = Path(args.out_root) / example_id / "decoder_query_trace"
        expected = out_dir / "decoder_query_trace" / "query_mapping.json"

        if args.skip_existing and expected.is_file():
            print(f"SKIP existing: {example_id}")
            continue

        out_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            "python",
            "-m",
            "experiments.detr_decoder_query_trace",
            "--image",
            image,
            "--prompt",
            prompt,
            "--out-dir",
            str(out_dir),
            "--device",
            args.device,
        ]

        print("\n==============================")
        print(example_id, "|", prompt)
        print(" ".join(cmd))
        print("==============================")

        try:
            subprocess.run(cmd, check=True, env=env)
        except subprocess.CalledProcessError as e:
            failures.append(
                {
                    "example_id": example_id,
                    "image": image,
                    "prompt": prompt,
                    "returncode": e.returncode,
                }
            )
            print(f"FAILED: {example_id}")

            if not args.continue_on_error:
                raise

    fail_path = Path(args.out_root) / "querytrace_failures.csv"
    if failures:
        with open(fail_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(failures[0].keys()))
            writer.writeheader()
            writer.writerows(failures)
        print(f"Saved failures: {fail_path}")


if __name__ == "__main__":
    main()