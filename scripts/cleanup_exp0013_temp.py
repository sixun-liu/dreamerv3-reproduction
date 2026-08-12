#!/usr/bin/env python3
"""Remove one stopped EXP-0013 run's data-disk temporary directories."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--temp-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wait-seconds", type=int, default=30)
    args = parser.parse_args()

    temp_root = args.temp_root.resolve()
    if not str(temp_root).startswith("/root/autodl-tmp/Runs/EXP-0013"):
        raise ValueError(f"Refusing unsafe cleanup root: {temp_root}")
    if temp_root.name != "tmp":
        raise ValueError(f"Expected run-scoped tmp directory: {temp_root}")
    deadline = time.monotonic() + args.wait_seconds
    active: list[dict[str, object]] = []
    while time.monotonic() < deadline:
        active = process_references(temp_root)
        if not active:
            break
        time.sleep(1)
    if active:
        raise RuntimeError(f"Run-scoped temporary directory is still active: {active}")
    before = sorted(str(path.resolve()) for path in temp_root.iterdir())
    for path in list(temp_root.iterdir()):
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink()
    after = sorted(str(path.resolve()) for path in temp_root.iterdir())
    result = {
        "schema_version": 1,
        "experiment_id": "EXP-0013",
        "temp_root": str(temp_root),
        "process_reference_count": len(active),
        "entries_before": before,
        "entries_after": after,
        "removed_count": len(before),
        "passed": not active and not after,
    }
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
