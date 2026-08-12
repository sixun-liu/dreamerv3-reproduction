#!/usr/bin/env python3
"""Generate or verify the EXP-0014 equal-budget single-environment config."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def render(source: dict, run_root: Path) -> str:
    config = yaml.safe_load(yaml.safe_dump(source, sort_keys=False))
    config["logdir"] = str(run_root / "envs1" / "train")
    config["script"] = "train"
    config["seed"] = 31415
    config["run"].update(
        debug=False,
        envs=1,
        steps=5040.0,
        train_ratio=32.0,
        log_every=10,
        report_every=120,
        save_every=300,
        save_at_end=True,
        from_checkpoint="",
    )
    return yaml.safe_dump(config, sort_keys=False, allow_unicode=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    source = yaml.safe_load(args.baseline.read_text(encoding="utf-8"))
    if source["task"] != "minecraft_diamond":
        raise ValueError("Baseline is not Minecraft Diamond")
    if source["run"]["envs"] != 1 or source["run"]["train_ratio"] != 32.0:
        raise ValueError("Unexpected EXP-0012 baseline")
    content = render(source, args.run_root.resolve())
    if args.check:
        if not args.output.is_file() or args.output.read_text(encoding="utf-8") != content:
            raise SystemExit(f"Generated config drift: {args.output}")
    else:
        if args.output.exists():
            raise FileExistsError(f"Refusing to overwrite: {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content, encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
