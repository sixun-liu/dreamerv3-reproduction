#!/usr/bin/env python3
"""Resolve the 2411f7d DreamerV3 config without launching a run."""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", dest="configs", action="append", default=[])
    args, overrides = parser.parse_known_args()

    sys.path.insert(0, str(args.runtime))
    import embodied
    from dreamerv3 import agent

    config = embodied.Config(agent.Agent.configs["defaults"])
    for name in args.configs:
        config = config.update(agent.Agent.configs[name])
    config = embodied.Flags(config).parse(overrides)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    config.save(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
