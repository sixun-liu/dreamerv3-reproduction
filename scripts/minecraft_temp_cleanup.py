#!/usr/bin/env python3
"""Safely remove one stopped Minecraft run's data-disk temporary tree."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import time
from pathlib import Path


RUNS_ROOT = Path("/root/autodl-tmp/Runs").resolve()
EXPERIMENT_PATTERN = re.compile(r"EXP-\d{4}")


def process_references(root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        try:
            cwd = (entry / "cwd").resolve()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        references = []
        if cwd == root or root in cwd.parents:
            references.append(str(cwd))
        try:
            descriptors = list((entry / "fd").iterdir())
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            descriptors = []
        for descriptor in descriptors:
            try:
                target = descriptor.resolve()
            except (FileNotFoundError, PermissionError, ProcessLookupError):
                continue
            if target == root or root in target.parents:
                references.append(str(target))
        if references:
            rows.append({"pid": pid, "references": sorted(set(references))})
    return rows


def validate_temp_root(temp_root: Path, experiment_id: str) -> Path:
    if not EXPERIMENT_PATTERN.fullmatch(experiment_id):
        raise ValueError(f"Invalid experiment ID: {experiment_id}")
    resolved = temp_root.resolve()
    try:
        relative = resolved.relative_to(RUNS_ROOT)
    except ValueError as error:
        raise ValueError(f"Temporary root is outside {RUNS_ROOT}: {resolved}") from error
    if len(relative.parts) < 3 or not relative.parts[0].startswith(
        f"{experiment_id}__"
    ):
        raise ValueError(f"Temporary root does not belong to {experiment_id}: {resolved}")
    if relative.parts[-2:] != ("work", "tmp"):
        raise ValueError(f"Expected a run-scoped work/tmp directory: {resolved}")
    return resolved


def cleanup_temp(
    temp_root: Path, experiment_id: str, wait_seconds: int
) -> dict[str, object]:
    resolved = validate_temp_root(temp_root, experiment_id)
    deadline = time.monotonic() + wait_seconds
    active: list[dict[str, object]] = []
    while time.monotonic() < deadline:
        active = process_references(resolved)
        if not active:
            break
        time.sleep(1)
    if active:
        raise RuntimeError(f"Run-scoped temporary directory is still active: {active}")
    before = sorted(str(path.resolve()) for path in resolved.iterdir())
    for path in list(resolved.iterdir()):
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink()
    after = sorted(str(path.resolve()) for path in resolved.iterdir())
    return {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "temp_root": str(resolved),
        "process_reference_count": len(active),
        "entries_before": before,
        "entries_after": after,
        "removed_count": len(before),
        "passed": not active and not after,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--temp-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wait-seconds", type=int, default=30)
    args = parser.parse_args()
    result = cleanup_temp(args.temp_root, args.experiment_id, args.wait_seconds)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
