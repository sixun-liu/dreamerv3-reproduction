#!/usr/bin/env python3
"""Create a verified XFS reflink clone of checkpoint and replay for recovery."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import cloudpickle


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_manifest(root: Path) -> dict:
    rows = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        stat = path.stat()
        rows.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size": stat.st_size,
                "sha256": sha256(path),
                "device": stat.st_dev,
                "inode": stat.st_ino,
            }
        )
    digest = hashlib.sha256()
    for row in rows:
        digest.update(
            f"{row['path']}\0{row['size']}\0{row['sha256']}\n".encode("utf-8")
        )
    return {
        "root": str(root.resolve()),
        "file_count": len(rows),
        "bytes": sum(row["size"] for row in rows),
        "tree_sha256": digest.hexdigest(),
        "files": rows,
    }


def indexed(manifest: dict) -> dict[str, dict]:
    return {row["path"]: row for row in manifest["files"]}


def compare_clone(source: dict, clone: dict) -> dict:
    source_rows = indexed(source)
    clone_rows = indexed(clone)
    same_paths = source_rows.keys() == clone_rows.keys()
    content_equal = same_paths and all(
        source_rows[name]["size"] == clone_rows[name]["size"]
        and source_rows[name]["sha256"] == clone_rows[name]["sha256"]
        for name in source_rows
    )
    distinct_inodes = same_paths and all(
        (source_rows[name]["device"], source_rows[name]["inode"])
        != (clone_rows[name]["device"], clone_rows[name]["inode"])
        for name in source_rows
    )
    return {
        "same_paths": same_paths,
        "content_equal": content_equal,
        "distinct_inodes": distinct_inodes,
        "passed": same_paths and content_equal and distinct_inodes,
    }


def latest_checkpoint(train_dir: Path) -> Path:
    latest = train_dir / "ckpt/latest"
    if not latest.is_file():
        raise ValueError(f"Missing checkpoint index: {latest}")
    name = latest.read_text(encoding="utf-8").strip()
    if not name or Path(name).name != name:
        raise ValueError(f"Invalid checkpoint name: {name!r}")
    checkpoint = train_dir / "ckpt" / name
    for filename in ("agent.pkl", "step.pkl", "replay.pkl", "done"):
        if not (checkpoint / filename).is_file():
            raise ValueError(f"Incomplete checkpoint: {checkpoint / filename}")
    return checkpoint


def prepare(
    source_train: Path,
    target_train: Path,
    expected_step: int,
    expected_checkpoint_sha256: str | None = None,
    expected_replay_sha256: str | None = None,
) -> dict:
    source_train = source_train.resolve()
    target_train = target_train.resolve()
    if target_train.exists():
        raise FileExistsError(f"Refusing to overwrite recovery target: {target_train}")
    source_checkpoint = latest_checkpoint(source_train)
    source_step = int(cloudpickle.loads((source_checkpoint / "step.pkl").read_bytes()))
    if source_step != expected_step:
        raise ValueError(f"Source checkpoint step {source_step} != {expected_step}")
    if not any((source_train / "replay").glob("*.npz")):
        raise ValueError("Source replay is empty")

    source = {
        "checkpoint": tree_manifest(source_train / "ckpt"),
        "replay": tree_manifest(source_train / "replay"),
    }
    expected = {
        "checkpoint": expected_checkpoint_sha256,
        "replay": expected_replay_sha256,
    }
    for name, digest in expected.items():
        if digest and source[name]["tree_sha256"] != digest:
            raise ValueError(
                f"Frozen {name} tree drift: {source[name]['tree_sha256']} != {digest}"
            )
    target_train.mkdir(parents=True)
    for name in ("ckpt", "replay"):
        subprocess.run(
            [
                "cp",
                "--archive",
                "--reflink=always",
                str(source_train / name),
                str(target_train / name),
            ],
            check=True,
        )
    clone = {
        "checkpoint": tree_manifest(target_train / "ckpt"),
        "replay": tree_manifest(target_train / "replay"),
    }
    comparisons = {
        name: compare_clone(source[name], clone[name]) for name in ("checkpoint", "replay")
    }
    result = {
        "schema_version": 1,
        "source_train": str(source_train),
        "target_train": str(target_train),
        "source_checkpoint": str(source_checkpoint),
        "source_step": source_step,
        "copy_method": "cp --archive --reflink=always",
        "frozen_expected_tree_sha256": expected,
        "source": source,
        "clone_before_training": clone,
        "comparisons": comparisons,
        "passed": all(row["passed"] for row in comparisons.values()),
    }
    if not result["passed"]:
        raise RuntimeError(f"Recovery clone verification failed: {comparisons}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-train", type=Path, required=True)
    parser.add_argument("--target-train", type=Path, required=True)
    parser.add_argument("--expected-step", type=int, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--expected-replay-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite: {args.output}")
    result = prepare(
        args.source_train,
        args.target_train,
        args.expected_step,
        args.expected_checkpoint_sha256,
        args.expected_replay_sha256,
    )
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: result[key] for key in ("source_step", "copy_method", "passed")}))


if __name__ == "__main__":
    main()
