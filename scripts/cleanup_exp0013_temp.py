#!/usr/bin/env python3
"""Compatibility CLI for EXP-0013 run-scoped temporary cleanup."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from minecraft_temp_cleanup import cleanup_temp, process_references

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--temp-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wait-seconds", type=int, default=30)
    args = parser.parse_args()

    result = cleanup_temp(args.temp_root, "EXP-0013", args.wait_seconds)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
