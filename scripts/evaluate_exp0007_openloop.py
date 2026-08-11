#!/usr/bin/env python3
"""Evaluate one EXP-0006 checkpoint on the frozen EXP-0007 panel."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from openloop_panel import sha256, verify_panel, write_deterministic_npz, write_json


MODEL_INPUT_KEYS = (
    "action",
    "height",
    "is_first",
    "is_last",
    "is_terminal",
    "orientations",
    "reward",
    "velocity",
)


def git_value(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True
    ).strip()


def flatten_tree(tree: dict, prefix: str = "") -> dict[str, np.ndarray]:
    output = {}
    for key, value in tree.items():
        name = f"{prefix}__{key}" if prefix else key
        if isinstance(value, dict):
            output.update(flatten_tree(value, name))
        else:
            output[name] = np.asarray(value)
    return output


def checkpoint_digest_file(run_dir: Path) -> dict[str, str]:
    result = {}
    for line in (run_dir / "checkpoint.sha256").read_text().splitlines():
        digest, filename = line.split(maxsplit=1)
        result[str(Path(filename))] = digest
    return result


def evaluate(args: argparse.Namespace) -> dict:
    verify_panel(
        args.panel, args.manifest, verify_sources=False, verify_hashes=False
    )
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if args.context != manifest["context_steps"]:
        raise ValueError("CLI context does not match panel manifest")
    if args.arm not in {"baseline", "e1", "p4"} or args.seed not in {0, 1}:
        raise ValueError((args.arm, args.seed))

    checkpoint = Path(
        (args.run_dir / "checkpoint_path.txt").read_text(encoding="utf-8").strip()
    )
    checkpoint_hashes = checkpoint_digest_file(args.run_dir)
    agent_pickle = checkpoint / "agent.pkl"
    actual_checkpoint_sha = sha256(agent_pickle)
    if checkpoint_hashes.get(str(agent_pickle)) != actual_checkpoint_sha:
        raise ValueError("Checkpoint agent.pkl SHA256 mismatch")

    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("MUJOCO_GL", "egl")
    sys.path.insert(0, str(args.runtime))
    import elements
    import ruamel.yaml as yaml
    from dreamerv3 import main as dreamer_main

    raw_config = yaml.YAML(typ="safe").load(
        (args.run_dir / "train" / "config.yaml").read_text(encoding="utf-8")
    )
    raw_config["logdir"] = str(
        args.output.parent / f"runtime-{args.arm}-s{args.seed:03d}"
    )
    raw_config["seed"] = args.eval_seed
    raw_config["replay_context"] = 0
    raw_config["jax"].update({
        "prealloc": False,
        "precompile": False,
        "enable_policy": False,
        "profiler": False,
    })
    config = elements.Config(raw_config)
    agent = dreamer_main.make_agent(config)
    loader = elements.Checkpoint()
    loader.agent = agent
    loader.load(checkpoint, keys=["agent"])

    with np.load(args.panel, allow_pickle=False) as panel:
        panel_count = len(panel["window_index"])
        if panel_count % args.batch_windows:
            raise ValueError(
                "Panel size must be divisible by batch-windows to avoid JIT padding"
            )
        collected: dict[str, list[np.ndarray]] = {}
        window_rows: list[np.ndarray] = []
        draw_rows: list[np.ndarray] = []
        for batch_index, start in enumerate(
            range(0, panel_count, args.batch_windows)
        ):
            stop = start + args.batch_windows
            windows = np.arange(start, stop, dtype=np.int32)
            batch = {
                key: np.repeat(np.asarray(panel[key][windows]), args.draws, axis=0)
                for key in MODEL_INPUT_KEYS
            }
            outputs = flatten_tree(
                agent.openloop(batch, args.context, seed_index=batch_index)
            )
            for key, value in outputs.items():
                if value.shape[0] != args.batch_windows * args.draws:
                    raise ValueError((key, value.shape))
                collected.setdefault(key, []).append(value)
            window_rows.append(np.repeat(windows, args.draws))
            draw_rows.append(np.tile(np.arange(args.draws, dtype=np.int16), len(windows)))

    arrays = {key: np.concatenate(values, axis=0) for key, values in collected.items()}
    arrays["window_index"] = np.concatenate(window_rows)
    arrays["draw_index"] = np.concatenate(draw_rows)
    nonfinite = {
        key: int((~np.isfinite(value)).sum())
        for key, value in arrays.items()
        if np.issubdtype(value.dtype, np.floating) and not np.isfinite(value).all()
    }
    if nonfinite:
        raise ValueError(f"Non-finite model outputs: {nonfinite}")
    write_deterministic_npz(args.output, arrays)

    runtime_commit = git_value(args.runtime, "rev-parse", "HEAD")
    runtime_status = git_value(args.runtime, "status", "--short")
    metadata = {
        "schema_version": 1,
        "experiment_id": "EXP-0007",
        "arm": args.arm,
        "train_seed": args.seed,
        "eval_seed": args.eval_seed,
        "draws_per_window": args.draws,
        "batch_windows": args.batch_windows,
        "context_steps": args.context,
        "prediction_steps": manifest["prediction_steps"],
        "panel_path": str(args.panel),
        "panel_sha256": manifest["panel_sha256"],
        "run_dir": str(args.run_dir),
        "train_config": str(args.run_dir / "train" / "config.yaml"),
        "train_config_sha256": sha256(args.run_dir / "train" / "config.yaml"),
        "checkpoint": str(checkpoint),
        "checkpoint_agent_sha256": actual_checkpoint_sha,
        "checkpoint_step": int(
            (args.run_dir / "checkpoint_step.txt").read_text().strip()
        ),
        "runtime_repo": str(args.runtime),
        "runtime_branch": git_value(args.runtime, "branch", "--show-current"),
        "runtime_commit": runtime_commit,
        "runtime_tracked_status": runtime_status,
        "output": str(args.output),
        "output_sha256": sha256(args.output),
        "row_count": len(arrays["window_index"]),
        "array_shapes": {key: list(value.shape) for key, value in arrays.items()},
        "cached_latents_used": False,
        "finite": True,
        "config_overrides": {
            "seed": args.eval_seed,
            "replay_context": 0,
            "jax.prealloc": False,
            "jax.precompile": False,
            "jax.enable_policy": False,
            "jax.profiler": False,
        },
    }
    write_json(args.metadata, metadata)
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--context", type=int, default=16)
    parser.add_argument("--draws", type=int, default=4)
    parser.add_argument("--batch-windows", type=int, default=16)
    parser.add_argument("--eval-seed", type=int, default=20260727)
    return parser.parse_args()


def main() -> None:
    metadata = evaluate(parse_args())
    print(json.dumps({
        "arm": metadata["arm"],
        "seed": metadata["train_seed"],
        "rows": metadata["row_count"],
        "output": metadata["output"],
        "sha256": metadata["output_sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()
