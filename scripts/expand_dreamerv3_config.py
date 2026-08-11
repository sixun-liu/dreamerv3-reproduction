#!/usr/bin/env python3
"""Expand a DreamerV3 runtime config without creating an environment or agent."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--generation", choices=("2024", "2026"), required=True)
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("overrides", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.overrides[:1] == ["--"]:
        args.overrides = args.overrides[1:]
    return args


def load_2024(runtime: Path, names: list[str], overrides: list[str]):
    sys.path.insert(0, str(runtime))
    os.chdir(runtime)
    import embodied  # type: ignore
    from dreamerv3.agent import Agent  # type: ignore

    config = embodied.Config(Agent.configs["defaults"])
    for name in names:
        config = config.update(Agent.configs[name])
    return embodied.Flags(config).parse(overrides)


def load_2026(runtime: Path, names: list[str], overrides: list[str]):
    sys.path.insert(0, str(runtime))
    os.chdir(runtime)
    import elements  # type: ignore
    import ruamel.yaml as yaml  # type: ignore

    path = runtime / "dreamerv3" / "configs.yaml"
    configs = yaml.YAML(typ="safe").load(path.read_text(encoding="utf-8"))
    config = elements.Config(configs["defaults"])
    for name in names:
        config = config.update(configs[name])
    return elements.Flags(config).parse(overrides)


def main() -> int:
    args = parse_args()
    runtime = args.runtime.resolve()
    output = args.output.resolve()
    if not (runtime / ".git").is_dir():
        raise SystemExit(f"Runtime must be a normal Git clone: {runtime}")
    if output.exists():
        raise SystemExit(f"Refusing to overwrite expanded config: {output}")

    loader = load_2024 if args.generation == "2024" else load_2026
    config = loader(runtime, args.configs, args.overrides)
    output.parent.mkdir(parents=True, exist_ok=True)
    config.save(output)
    print(json.dumps({
        "runtime": str(runtime),
        "generation": args.generation,
        "configs": args.configs,
        "overrides": args.overrides,
        "output": str(output),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
