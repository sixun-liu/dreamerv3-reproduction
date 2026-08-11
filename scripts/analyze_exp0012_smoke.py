#!/usr/bin/env python3
"""Apply the frozen EXP-0012 model-smoke resource gate."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--integrity", type=Path, required=True)
    parser.add_argument("--gpu-csv", type=Path, required=True)
    parser.add_argument("--wall-seconds", type=int, required=True)
    parser.add_argument("--output-bytes", type=int, required=True)
    parser.add_argument("--disk-free-bytes", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    integrity = json.loads(args.integrity.read_text(encoding="utf-8"))
    with args.gpu_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    memory = [float(row["memory_used_mib"]) for row in rows if row["memory_used_mib"]]
    utilization = [float(row["gpu_util_percent"]) for row in rows if row["gpu_util_percent"]]
    projected_wall_seconds = args.wall_seconds * 100000 / 4096
    projected_output_bytes = args.output_bytes * 100000 / 4096
    gates = {
        "integrity_passed": bool(integrity.get("passed")),
        "gpu_samples_present": bool(memory),
        "peak_vram_mib_le_30500": bool(memory) and max(memory) <= 30500,
        "projected_100k_wall_le_12h": projected_wall_seconds <= 12 * 3600,
        "projected_output_plus_5gib_fits": (
            projected_output_bytes + 5 * 1024**3 < args.disk_free_bytes
        ),
    }
    result = {
        "schema_version": 1,
        "experiment_id": "EXP-0012",
        "stage": "smoke",
        "wall_seconds": args.wall_seconds,
        "output_bytes": args.output_bytes,
        "disk_free_bytes": args.disk_free_bytes,
        "gpu_samples": len(memory),
        "peak_vram_mib": max(memory) if memory else None,
        "mean_gpu_util_percent": sum(utilization) / len(utilization) if utilization else None,
        "projected_100k_wall_seconds_conservative": projected_wall_seconds,
        "projected_100k_output_bytes": projected_output_bytes,
        "gates": gates,
        "formal_gate": all(gates.values()),
    }
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
