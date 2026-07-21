#!/usr/bin/env python3
"""Resolve DreamerV3 named configs and CLI overrides without launching a run."""

import argparse
from pathlib import Path

import elements
import ruamel.yaml as yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", dest="configs", action="append", default=[])
    args, overrides = parser.parse_known_args()

    source = args.runtime / "dreamerv3" / "configs.yaml"
    configs = yaml.YAML(typ="safe").load(source.read_text(encoding="utf-8"))
    config = elements.Config(configs["defaults"])
    for name in args.configs:
        config = config.update(configs[name])
    config = elements.Flags(config).parse(overrides)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    config.save(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
