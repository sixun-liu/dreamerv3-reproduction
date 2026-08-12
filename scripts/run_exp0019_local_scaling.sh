#!/usr/bin/env bash
set -euo pipefail

readonly CONTROL=/root/autodl-tmp/Code/DreamerV3/dreamerv3-reproduction
readonly PYTHON=/root/autodl-tmp/Envs/dv3-minecraft-2026/bin/python
readonly RUN_ROOT=/root/autodl-tmp/Runs/EXP-0019__minecraft-local-scaling__s000__100k-to-105040-env__20260812T093000Z
readonly REVIEW=/root/autodl-tmp/Artifacts/dreamerv3/review/EXP-0019-minecraft-envs4-envs5-local-scaling

if [[ -e "${RUN_ROOT}" || -e "${RUN_ROOT}.started" ]]; then
  echo "Refusing duplicate EXP-0019 launch" >&2
  exit 20
fi
printf '{"experiment_id":"EXP-0019","started_at":"%s","pid":%d}\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$$" > "${RUN_ROOT}.started"

"${CONTROL}/scripts/run_exp0019_arm.sh" 4
"${CONTROL}/scripts/run_exp0019_arm.sh" 5
"${PYTHON}" "${CONTROL}/scripts/analyze_exp0019_local_scaling.py" \
  --run-root "${RUN_ROOT}" --output "${RUN_ROOT}/comparison.json" \
  > "${RUN_ROOT}/comparison_stdout.log" 2>&1
"${PYTHON}" "${CONTROL}/scripts/build_exp0019_review.py" \
  --comparison "${RUN_ROOT}/comparison.json" --output-dir "${REVIEW}" \
  > "${RUN_ROOT}/review_stdout.log" 2>&1
selected=$("${PYTHON}" -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["comparison"]["selected_environment_count"])' \
  "${RUN_ROOT}/comparison.json")
printf \
  '{"experiment_id":"EXP-0019","completed_at":"%s","selected_environment_count":%s,"comparison":"%s","review":"%s"}\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${selected}" \
  "${RUN_ROOT}/comparison.json" "${REVIEW}" > "${RUN_ROOT}/.completed"
