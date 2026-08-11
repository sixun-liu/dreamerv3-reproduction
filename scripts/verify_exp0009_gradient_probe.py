#!/usr/bin/env python3
"""Verify the plumbing of one EXP-0009 gradient probe without score claims."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import cloudpickle
import yaml


ERROR_MARKERS = ("Traceback", "AssertionError", "Out of memory", "CUDA_ERROR", "ResourceExhaustedError")
ALLOWED_NONFINITE_KEYS = {
    "replay/replay_ratio", "replay/insert_wait_avg", "replay/insert_wait_frac",
    "replay/sample_wait_avg", "replay/sample_wait_frac",
}


def sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open('rb') as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b''):
      digest.update(chunk)
  return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict]:
  with path.open(encoding='utf-8') as handle:
    return [json.loads(line) for line in handle if line.strip()]


def unexpected_nonfinite(rows: list[dict]) -> list[dict]:
  result = []
  for index, row in enumerate(rows):
    for key, value in row.items():
      if not isinstance(value, float) or math.isfinite(value):
        continue
      empty_balance = (
          key.startswith(('train/', 'report/')) and
          ('/constats/' in key or '/rewstats/' in key) and
          key.rsplit('/', 1)[-1] in {'neg_acc', 'neg_loss', 'pos_acc', 'pos_loss'})
      if key not in ALLOWED_NONFINITE_KEYS and not empty_balance and not (
          key.startswith('timer/') and key.endswith('/min')):
        result.append({'row': index, 'key': key, 'value': str(value)})
  return result


def verify(run_dir: Path, frozen_config: Path, expected_step: int) -> dict:
  train_dir = run_dir / 'train'
  generated = train_dir / 'config.yaml'
  checkpoint = train_dir / 'checkpoint.ckpt'
  metrics_path = train_dir / 'metrics.jsonl'
  stdout_path = run_dir / 'train_stdout.log'
  required = (frozen_config, generated, checkpoint, metrics_path, stdout_path)
  missing = [str(path) for path in required if not path.is_file()]
  if missing:
    raise ValueError(f'Missing required files: {missing}')
  frozen = yaml.safe_load(frozen_config.read_text(encoding='utf-8'))
  actual = yaml.safe_load(generated.read_text(encoding='utf-8'))
  metrics = load_jsonl(metrics_path)
  checkpoint_data = cloudpickle.loads(checkpoint.read_bytes())
  stdout = stdout_path.read_text(encoding='utf-8', errors='replace')
  nonfinite = unexpected_nonfinite(metrics)
  error_markers = [marker for marker in ERROR_MARKERS if marker in stdout]
  result = {
      'config_semantically_equal': frozen == actual,
      'checkpoint_step': int(checkpoint_data['step']),
      'checkpoint_step_matches': int(checkpoint_data['step']) == expected_step,
      'checkpoint_sha256': sha256(checkpoint),
      'metric_rows': len(metrics),
      'metrics_unexpected_nonfinite': nonfinite,
      'error_markers': error_markers,
  }
  result['passed'] = bool(
      result['config_semantically_equal'] and result['checkpoint_step_matches'] and
      metrics and not nonfinite and not error_markers)
  return result


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument('--run-dir', type=Path, required=True)
  parser.add_argument('--frozen-config', type=Path, required=True)
  parser.add_argument('--expected-step', type=int, required=True)
  parser.add_argument('--output', type=Path, required=True)
  args = parser.parse_args()
  result = verify(args.run_dir, args.frozen_config, args.expected_step)
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
  print(json.dumps(result, indent=2, ensure_ascii=False))
  if not result['passed']:
    raise SystemExit(1)


if __name__ == '__main__':
  main()
