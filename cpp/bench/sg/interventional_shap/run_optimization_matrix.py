# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Checkpointed scaling and pattern-tile matrix; runs GPU jobs sequentially."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--library", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=["scaling", "tiles"], required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    if args.stage == "scaling":
        for rows in [1, 100, 10000]:
            for background in [1, 10, 100, 1000]:
                jobs.append(
                    (
                        f"scaling-n{rows}-b{background}",
                        ["--rows", str(rows), "--background", str(background)],
                    )
                )
    else:
        for tile in [8, 64, 256]:
            jobs.append(
                (
                    f"tiles-{tile}",
                    [
                        "--rows",
                        "1000",
                        "--background",
                        "100",
                        "--tile",
                        str(tile),
                        "--variants",
                        "pattern_background",
                        "pattern_foreground",
                        "pattern_sparse",
                        "pattern_dense",
                        "pattern_adaptive",
                    ],
                )
            )
    for name, options in jobs:
        output = args.output_dir / f"{name}.json"
        if output.exists():
            data = json.loads(output.read_text())
            protocol = data["protocol"]
            records = data["results"]
            if len(records) == len(protocol["models"]) * len(
                protocol["variants"]
            ) and all(
                r.get("status") == "unsupported"
                or len(r["cold"]) == protocol["repeats"]
                for r in records
            ):
                print("SKIP", name, flush=True)
                continue
        cmd = [
            sys.executable,
            str(Path(__file__).with_name("benchmark_optimizations.py")),
            "--cache",
            args.cache,
            "--library",
            args.library,
            "--output",
            str(output),
            "--repeats",
            "3",
            *options,
        ]
        print("START", name, flush=True)
        subprocess.run(cmd, check=True)
        print("DONE", name, flush=True)


if __name__ == "__main__":
    main()
