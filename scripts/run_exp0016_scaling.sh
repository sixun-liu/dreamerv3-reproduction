#!/usr/bin/env bash
set -euo pipefail

readonly CONTROL=/root/autodl-tmp/Code/DreamerV3/dreamerv3-reproduction
readonly PYTHON=/root/autodl-tmp/Envs/dv3-minecraft-2026/bin/python
readonly RUN_ROOT=/root/autodl-tmp/Runs/EXP-0016__minecraft-recovery-scaling__s000__100k-to-105040-env__20260812T073000Z

"${CONTROL}/scripts/run_exp0016_arm.sh" 4
if "${PYTHON}" -c \
    'import json,sys; assert json.load(open(sys.argv[1]))["comparison"]["candidate_promoted"]' \
    "${RUN_ROOT}/envs4_vs_envs2.json"; then
  "${CONTROL}/scripts/run_exp0016_arm.sh" 8
else
  printf \
    '{"experiment_id":"EXP-0016","completed_at":"%s","selected_environment_count":2,"envs8_skipped":true,"reason":"envs4_not_promoted"}\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${RUN_ROOT}/.completed"
fi

if [[ -f "${RUN_ROOT}/envs8_vs_envs4.json" ]]; then
  selected=$("${PYTHON}" -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["comparison"]["selected_environment_count"])' \
    "${RUN_ROOT}/envs8_vs_envs4.json")
  printf \
    '{"experiment_id":"EXP-0016","completed_at":"%s","selected_environment_count":%s,"envs8_skipped":false}\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${selected}" > "${RUN_ROOT}/.completed"
fi
