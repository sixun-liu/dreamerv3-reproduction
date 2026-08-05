#!/usr/bin/env bash
set -euo pipefail

readonly CONTROL=/root/autodl-tmp/dreamerv3-reproduction
readonly PYTHON=/root/autodl-tmp/envs/dv3-2411/bin/python
readonly TAG=EXP-0008__cheetah-run__five-seed__500k-env__20260805T171000Z
readonly MATRIX=/root/autodl-tmp/runs/${TAG}
readonly REFERENCE=/root/autodl-tmp/dreamerv3/scores/dmc_proprio-dreamerv3.json.gz
readonly OUTPUT=/root/autodl-tmp/artifacts/dreamerv3/EXP-0008
readonly REVIEW=/root/autodl-tmp/artifacts/dreamerv3/review/EXP-0008-cheetah-five-seed

"${PYTHON}" "${CONTROL}/scripts/analyze_exp0008_cheetah.py" \
  --matrix-root "${MATRIX}" \
  --reference "${REFERENCE}" \
  --output-dir "${OUTPUT}" \
  --review-dir "${REVIEW}"

"${PYTHON}" "${CONTROL}/scripts/verify_exp0008_independent.py" \
  --matrix-root "${MATRIX}" \
  --reference "${REFERENCE}" \
  --summary "${OUTPUT}/summary.json" \
  --output "${OUTPUT}/independent_verification.json"
