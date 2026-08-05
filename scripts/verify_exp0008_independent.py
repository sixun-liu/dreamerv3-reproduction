#!/usr/bin/env python3
"""Independently recompute EXP-0008 headline values from raw run files."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import statistics
from pathlib import Path

import cloudpickle


REFERENCE_SHA256 = "8182860a8a56dc56836c319fde9b941376621e1e0d474141c7d174ab833cc7f4"


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def close(left: float, right: float, tolerance: float = 1e-9) -> bool:
    return math.isclose(left, right, rel_tol=tolerance, abs_tol=tolerance)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-root", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if file_hash(args.reference) != REFERENCE_SHA256:
        raise ValueError("Reference hash drift")
    with gzip.open(args.reference, "rt", encoding="utf-8") as handle:
        official_rows = [row for row in json.load(handle) if row["task"] == "dmc_cheetah_run"]
    official_rows.sort(key=lambda row: int(row["seed"]))
    official_finals = [statistics.fmean(map(float, row["ys"][-3:])) for row in official_rows]

    local_finals = []
    checkpoint_hashes = []
    completion = []
    for seed in range(5):
        run_dir = args.matrix_root / f"s{seed:03d}"
        score_rows = [
            json.loads(line)
            for line in (run_dir / "train" / "scores.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        final_scores = [
            float(row["episode/score"])
            for row in score_rows
            if 470_000 < float(row["step"]) <= 500_000
        ]
        if not final_scores or not all(math.isfinite(value) for value in final_scores):
            raise ValueError(f"Seed {seed} final window is empty or non-finite")
        local_finals.append(statistics.fmean(final_scores))
        checkpoint = run_dir / "train" / "checkpoint.ckpt"
        checkpoint_data = cloudpickle.loads(checkpoint.read_bytes())
        if int(checkpoint_data["step"]) != 250_000:
            raise ValueError(f"Seed {seed} checkpoint step mismatch")
        checkpoint_hashes.append(file_hash(checkpoint))
        completion.append(Path(str(run_dir) + ".completed").is_file())

    expected = json.loads(args.summary.read_text(encoding="utf-8"))
    local_mean = statistics.fmean(local_finals)
    official_mean = statistics.fmean(official_finals)
    official_range = [min(official_finals), max(official_finals)]
    primary_gate = all(completion) and official_range[0] <= local_mean <= official_range[1]
    comparisons = {
        "official_seed_finals": all(
            close(left, right)
            for left, right in zip(
                official_finals,
                expected["official"]["seed_final_three_curve_point_means"],
            )
        ),
        "official_mean": close(official_mean, expected["official"]["mean"]),
        "local_seed_finals": all(
            close(left, right)
            for left, right in zip(local_finals, expected["local"]["seed_final_window_means"])
        ),
        "local_mean": close(local_mean, expected["local"]["mean"]),
        "checkpoint_hashes": checkpoint_hashes
        == [row["checkpoint_sha256"] for row in expected["per_seed"]],
        "primary_gate": primary_gate == expected["gates"]["primary_replication_gate"],
    }
    output = {
        "experiment_id": "EXP-0008",
        "implementation": "independent raw-file recomputation; does not import the primary analyzer",
        "official_seed_finals": official_finals,
        "official_mean": official_mean,
        "local_seed_finals": local_finals,
        "local_mean": local_mean,
        "checkpoint_hashes": checkpoint_hashes,
        "comparisons": comparisons,
        "passed": all(comparisons.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(output, indent=2, ensure_ascii=False))
    if not output["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
