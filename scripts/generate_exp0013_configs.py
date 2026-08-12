#!/usr/bin/env python3
"""Generate or verify EXP-0013 expanded runtime configs from EXP-0012."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


ENVS = (2, 4, 8)


def render(source: dict, run_root: Path, envs: int) -> str:
    config = yaml.safe_load(yaml.safe_dump(source, sort_keys=False))
    config["logdir"] = str(run_root / f"envs{envs}" / "train")
    config["script"] = "train"
    config["seed"] = 31415
    config["run"]["debug"] = False
    config["run"]["envs"] = envs
    config["run"]["steps"] = 5040.0
    config["run"]["train_ratio"] = 32.0
    config["run"]["log_every"] = 10
    config["run"]["report_every"] = 120
    config["run"]["save_every"] = 300
    config["run"]["save_at_end"] = True
    config["run"]["from_checkpoint"] = ""
    return yaml.safe_dump(config, sort_keys=False, allow_unicode=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    source = yaml.safe_load(args.baseline.read_text(encoding="utf-8"))
    if source["task"] != "minecraft_diamond":
        raise ValueError("Baseline is not Minecraft Diamond")
    if source["run"]["envs"] != 1 or source["run"]["train_ratio"] != 32.0:
        raise ValueError("Unexpected EXP-0012 baseline")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for envs in ENVS:
        path = args.output_dir / f"exp0013_minecraft_envs{envs}_s31415_5040_env.yaml"
        content = render(source, args.run_root.resolve(), envs)
        if args.check:
            if not path.is_file() or path.read_text(encoding="utf-8") != content:
                raise SystemExit(f"Generated config drift: {path}")
        else:
            if path.exists():
                raise FileExistsError(f"Refusing to overwrite: {path}")
            path.write_text(content, encoding="utf-8")
        print(path)


if __name__ == "__main__":
    main()
