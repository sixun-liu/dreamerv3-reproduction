#!/usr/bin/env python3
"""Build and verify the shared replay panel used by EXP-0007."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import tempfile
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ARMS = ("baseline", "e1", "p4")
SEEDS = (0, 1)
ARRAY_KEYS = (
    "action",
    "height",
    "is_first",
    "is_last",
    "is_terminal",
    "orientations",
    "reward",
    "stepid",
    "velocity",
)
BOUNDARY_KEYS = ("is_first", "is_last", "is_terminal")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    path.write_text(text, encoding="utf-8")


def write_deterministic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Write an NPZ with stable member order, timestamps, and permissions."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    os.close(fd)
    temporary_path = Path(temporary)
    try:
        with zipfile.ZipFile(
            temporary_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
        ) as archive:
            for key in sorted(arrays):
                stream = io.BytesIO()
                np.lib.format.write_array(
                    stream, np.asanyarray(arrays[key]), allow_pickle=False
                )
                info = zipfile.ZipInfo(f"{key}.npy", (1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o600 << 16
                archive.writestr(
                    info,
                    stream.getvalue(),
                    compress_type=zipfile.ZIP_DEFLATED,
                    compresslevel=6,
                )
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def source_specs(matrix_root: Path) -> list[dict]:
    specs = []
    for arm in ARMS:
        for seed in SEEDS:
            replay = matrix_root / arm / f"s{seed:03d}" / "train" / "replay"
            if not replay.is_dir():
                raise FileNotFoundError(replay)
            specs.append({"arm": arm, "seed": seed, "replay": replay})
    return specs


def valid_starts(path: Path, window_length: int) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        missing = set(ARRAY_KEYS) - set(data.files)
        if missing:
            raise ValueError(f"{path}: missing keys {sorted(missing)}")
        lengths = {key: len(data[key]) for key in ARRAY_KEYS}
        if len(set(lengths.values())) != 1:
            raise ValueError(f"{path}: inconsistent array lengths {lengths}")
        length = next(iter(lengths.values()))
        if length < window_length:
            return np.empty((0,), np.int64)
        boundary = np.logical_or.reduce([
            np.asarray(data[key], bool) for key in BOUNDARY_KEYS
        ])
    counts = np.convolve(
        boundary.astype(np.int16), np.ones(window_length, np.int16), mode="valid"
    )
    return np.flatnonzero(counts == 0).astype(np.int64)


def select_candidates(
    candidates: list[tuple[Path, int]],
    count: int,
    window_length: int,
    rng: np.random.Generator,
) -> list[tuple[Path, int]]:
    accepted: list[tuple[Path, int]] = []
    by_file: dict[Path, list[int]] = defaultdict(list)
    for index in rng.permutation(len(candidates)):
        path, start = candidates[int(index)]
        if any(abs(start - other) < window_length for other in by_file[path]):
            continue
        accepted.append((path, start))
        by_file[path].append(start)
        if len(accepted) == count:
            return accepted
    raise ValueError(
        f"Only {len(accepted)} non-overlapping windows available; requested {count}"
    )


def build_panel(
    matrix_root: Path,
    output: Path,
    manifest_path: Path,
    *,
    context: int,
    horizon: int,
    windows_per_source: int,
    seed: int,
) -> dict:
    window_length = context + horizon
    if context < 1 or horizon < 1:
        raise ValueError((context, horizon))

    arrays: dict[str, list[np.ndarray]] = {key: [] for key in ARRAY_KEYS}
    source_indices: list[int] = []
    records: list[dict] = []
    source_rows: list[dict] = []
    used_files: dict[Path, str] = {}

    for source_index, spec in enumerate(source_specs(matrix_root)):
        candidates: list[tuple[Path, int]] = []
        chunks = sorted(spec["replay"].glob("*.npz"))
        if not chunks:
            raise ValueError(f"No replay chunks under {spec['replay']}")
        for chunk in chunks:
            candidates.extend(
                (chunk, int(start)) for start in valid_starts(chunk, window_length)
            )
        rng = np.random.default_rng(np.random.SeedSequence([seed, source_index]))
        selected = select_candidates(
            candidates, windows_per_source, window_length, rng
        )
        source_rows.append({
            "source_index": source_index,
            "arm": spec["arm"],
            "seed": spec["seed"],
            "replay_root": str(spec["replay"]),
            "chunk_count": len(chunks),
            "valid_candidate_count": len(candidates),
            "selected_count": len(selected),
        })
        for local_index, (chunk, start) in enumerate(selected):
            with np.load(chunk, allow_pickle=False) as data:
                for key in ARRAY_KEYS:
                    arrays[key].append(
                        np.array(data[key][start:start + window_length], copy=True)
                    )
            used_files.setdefault(chunk, sha256(chunk))
            window_index = len(source_indices)
            source_indices.append(source_index)
            records.append({
                "window_index": window_index,
                "source_index": source_index,
                "source_local_index": local_index,
                "arm": spec["arm"],
                "seed": spec["seed"],
                "chunk": str(chunk),
                "chunk_sha256": used_files[chunk],
                "start": start,
                "stop": start + window_length,
            })

    output_arrays = {
        key: np.stack(values, axis=0) for key, values in arrays.items()
    }
    output_arrays["source_index"] = np.asarray(source_indices, np.int16)
    output_arrays["window_index"] = np.arange(len(source_indices), dtype=np.int32)
    write_deterministic_npz(output, output_arrays)

    manifest = {
        "schema_version": 1,
        "experiment_id": "EXP-0007",
        "purpose": "Shared finite-context open-loop prediction panel",
        "matrix_root": str(matrix_root),
        "panel_path": str(output),
        "panel_sha256": sha256(output),
        "builder": str(Path(__file__).resolve()),
        "numpy_version": np.__version__,
        "sampling_seed": seed,
        "context_steps": context,
        "prediction_steps": horizon,
        "window_length": window_length,
        "windows_per_source": windows_per_source,
        "window_count": len(source_indices),
        "array_keys": sorted(output_arrays),
        "forbidden_replay_keys": ["dyn/deter", "dyn/stoch"],
        "boundary_policy": (
            "Reject any window containing is_first, is_last, or is_terminal; "
            "reset latent state at panel window start and warm up on raw observations."
        ),
        "action_alignment": (
            "Replay action[t] predicts observation and reward[t+1]; context ends "
            "at observation[context-1]."
        ),
        "sources": source_rows,
        "used_files": [
            {"path": str(path), "sha256": digest}
            for path, digest in sorted(used_files.items(), key=lambda item: str(item[0]))
        ],
        "windows": records,
    }
    write_json(manifest_path, manifest)
    verify_panel(output, manifest_path, verify_sources=True, verify_hashes=True)
    return manifest


def verify_panel(
    panel_path: Path,
    manifest_path: Path,
    *,
    verify_sources: bool,
    verify_hashes: bool,
) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if Path(manifest["panel_path"]) != panel_path:
        raise ValueError("Manifest panel path does not match CLI panel path")
    if sha256(panel_path) != manifest["panel_sha256"]:
        raise ValueError("Panel SHA256 mismatch")

    with np.load(panel_path, allow_pickle=False) as panel:
        if any(key.startswith("dyn/") for key in panel.files):
            raise ValueError("Panel contains forbidden cached latent state")
        missing = (set(ARRAY_KEYS) | {"source_index", "window_index"}) - set(panel.files)
        if missing:
            raise ValueError(f"Panel missing keys {sorted(missing)}")
        count = manifest["window_count"]
        length = manifest["window_length"]
        for key in ARRAY_KEYS:
            if panel[key].shape[:2] != (count, length):
                raise ValueError((key, panel[key].shape, count, length))
        boundary = np.logical_or.reduce([
            np.asarray(panel[key], bool) for key in BOUNDARY_KEYS
        ])
        if boundary.any():
            raise ValueError("Panel contains an episode boundary")
        counts = Counter(np.asarray(panel["source_index"], int).tolist())
        expected = manifest["windows_per_source"]
        if set(counts.values()) != {expected} or len(counts) != len(manifest["sources"]):
            raise ValueError(f"Unbalanced panel sources: {counts}")

        if verify_sources:
            grouped: dict[Path, list[dict]] = defaultdict(list)
            for row in manifest["windows"]:
                grouped[Path(row["chunk"])].append(row)
            for path, rows in grouped.items():
                if verify_hashes and sha256(path) != rows[0]["chunk_sha256"]:
                    raise ValueError(f"Replay chunk SHA256 mismatch: {path}")
                with np.load(path, allow_pickle=False) as source:
                    for row in rows:
                        index = row["window_index"]
                        start, stop = row["start"], row["stop"]
                        for key in ARRAY_KEYS:
                            if not np.array_equal(panel[key][index], source[key][start:stop]):
                                raise ValueError(
                                    f"Source mismatch: window={index} key={key}"
                                )
    return {
        "panel_sha256": manifest["panel_sha256"],
        "window_count": manifest["window_count"],
        "window_length": manifest["window_length"],
        "source_count": len(manifest["sources"]),
        "source_verified": verify_sources,
        "hashes_verified": verify_hashes,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--matrix-root", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--manifest", type=Path, required=True)
    build.add_argument("--context", type=int, default=16)
    build.add_argument("--horizon", type=int, default=15)
    build.add_argument("--windows-per-source", type=int, default=96)
    build.add_argument("--seed", type=int, default=20260727)

    verify = subparsers.add_parser("verify")
    verify.add_argument("--panel", type=Path, required=True)
    verify.add_argument("--manifest", type=Path, required=True)
    verify.add_argument("--source", action="store_true")
    verify.add_argument("--hashes", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "build":
        result = build_panel(
            args.matrix_root,
            args.output,
            args.manifest,
            context=args.context,
            horizon=args.horizon,
            windows_per_source=args.windows_per_source,
            seed=args.seed,
        )
        print(json.dumps({
            "panel": result["panel_path"],
            "sha256": result["panel_sha256"],
            "windows": result["window_count"],
        }, indent=2))
    else:
        print(json.dumps(verify_panel(
            args.panel,
            args.manifest,
            verify_sources=args.source,
            verify_hashes=args.hashes,
        ), indent=2))


if __name__ == "__main__":
    main()
