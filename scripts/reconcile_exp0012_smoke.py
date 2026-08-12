#!/usr/bin/env python3
"""Reconcile the retained EXP-0012 smoke after an instrumentation-only failure."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from analyze_exp0012_smoke import compute_smoke_gate
from verify_exp0012_run import verify


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def git_output(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True
    ).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--frozen-config", type=Path, required=True)
    parser.add_argument("--control-repo", type=Path, required=True)
    parser.add_argument("--expected-step", type=int, default=4096)
    parser.add_argument("--driver-step-quantum", type=int, default=10)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    control_repo = args.control_repo.resolve()
    original_integrity_path = run_dir / "integrity_smoke.json"
    original_failure_path = run_dir / ".smoke.failed"
    stdout_path = run_dir / "train_smoke_stdout.log"
    gpu_path = run_dir / "resource_smoke_gpu.csv"
    run_freeze_path = run_dir / ".freeze"
    outputs = {
        "integrity": run_dir / "integrity_smoke_reconciled.json",
        "gate": run_dir / "gate_smoke.json",
        "reconciliation": run_dir / "smoke_reconciliation.json",
        "stage_completed": run_dir / ".smoke.completed",
        "run_completed": run_dir / ".completed",
        "external_completed": Path(str(run_dir) + ".completed"),
    }
    if any(path.exists() for path in outputs.values()):
        raise SystemExit("Refusing to overwrite an existing reconciliation output")
    required = (
        original_integrity_path,
        original_failure_path,
        stdout_path,
        gpu_path,
        run_freeze_path,
        args.frozen_config,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit(f"Missing retained smoke evidence: {missing}")
    if git_output(control_repo, "status", "--porcelain"):
        raise SystemExit("Control repo must be clean before reconciliation")

    original_integrity = json.loads(original_integrity_path.read_text(encoding="utf-8"))
    original_failure = json.loads(original_failure_path.read_text(encoding="utf-8"))
    run_freeze = json.loads(run_freeze_path.read_text(encoding="utf-8"))
    if original_integrity.get("passed"):
        raise SystemExit("Original integrity already passed; reconciliation is not applicable")
    if original_failure.get("phase") != "integrity":
        raise SystemExit("Original failure was not instrumentation-only")
    if original_integrity.get("checkpoint_step") != 4100:
        raise SystemExit("Unexpected retained checkpoint step")
    failed_checks = [
        key
        for key, value in original_integrity.items()
        if key.startswith("checkpoint_step_") and value is False
    ]
    if failed_checks != ["checkpoint_step_matches"]:
        raise SystemExit(f"Unexpected original checkpoint failures: {failed_checks}")

    reconciled = verify(
        run_dir,
        args.frozen_config,
        args.expected_step,
        args.driver_step_quantum,
        stdout_path,
        require_scores=False,
    )
    if not reconciled["passed"]:
        raise SystemExit("Reconciled integrity did not pass")

    with gpu_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    memory = [float(row["memory_used_mib"]) for row in rows if row["memory_used_mib"]]
    utilization = [
        float(row["gpu_util_percent"])
        for row in rows
        if row["gpu_util_percent"]
    ]
    output_bytes = sum(
        path.stat().st_size for path in run_dir.rglob("*") if path.is_file()
    )
    disk_free_bytes = shutil.disk_usage(run_dir).free
    gate = compute_smoke_gate(
        reconciled,
        memory,
        utilization,
        int(original_failure["wall_seconds"]),
        output_bytes,
        disk_free_bytes,
    )
    if not gate["formal_gate"]:
        raise SystemExit("Reconciled smoke did not pass the frozen resource gate")
    write_json(outputs["integrity"], reconciled)
    write_json(outputs["gate"], gate)

    control_commit = git_output(control_repo, "rev-parse", "HEAD")
    completed_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    reconciliation = {
        "schema_version": 1,
        "experiment_id": "EXP-0012",
        "stage": "smoke",
        "reconciled_at": completed_at,
        "reason": (
            "The frozen runtime advances the single-environment driver in 10-step "
            "chunks, so a 4096-step request naturally stops at step 4100."
        ),
        "scientific_inputs_changed": False,
        "control_commit": control_commit,
        "runtime_commit": run_freeze["runtime_commit"],
        "original_control_commit": run_freeze["control_commit"],
        "original_failure_retained": True,
        "original_integrity_sha256": sha256(original_integrity_path),
        "original_failure_sha256": sha256(original_failure_path),
        "reconciled_integrity_sha256": sha256(outputs["integrity"]),
        "gate_sha256": sha256(outputs["gate"]),
        "requested_environment_steps": args.expected_step,
        "driver_step_quantum": args.driver_step_quantum,
        "checkpoint_step": reconciled["checkpoint_step"],
        "formal_gate": True,
    }
    write_json(outputs["reconciliation"], reconciliation)

    completion = {
        "experiment_id": "EXP-0012",
        "stage": "smoke",
        "completed_at": completed_at,
        "exit_code": 0,
        "reconciled": True,
        "original_failure_retained": True,
        "requested_environment_steps": args.expected_step,
        "checkpoint_step": reconciled["checkpoint_step"],
        "environment_steps": reconciled["checkpoint_step"],
        "driver_step_quantum": args.driver_step_quantum,
        "wall_seconds": int(original_failure["wall_seconds"]),
        "output_bytes": output_bytes,
    }
    for key in ("stage_completed", "run_completed", "external_completed"):
        write_json(outputs[key], completion)
    print(json.dumps({"reconciliation": reconciliation, "gate": gate}, indent=2))


if __name__ == "__main__":
    main()
