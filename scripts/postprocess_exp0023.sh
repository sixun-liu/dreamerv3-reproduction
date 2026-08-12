#!/usr/bin/env bash
set -euo pipefail

readonly CONTROL=/root/autodl-tmp/Code/DreamerV3/dreamerv3-reproduction
readonly PYTHON=/root/autodl-tmp/Envs/dv3-minecraft-2026/bin/python
readonly EXPERIMENT=EXP-0023
readonly TRAIN=/root/autodl-tmp/Runs/EXP-0023__minecraft-diamond__s000__200k-to-500k-env__20260812T123000Z
readonly EVAL=/root/autodl-tmp/Runs/EXP-0023__minecraft-parallel-eval__s10000__3eps__20260812T123000Z
readonly SOURCE=/root/autodl-tmp/Runs/EXP-0020__minecraft-diamond__s000__100k-to-200k-env__20260812T100000Z
readonly BASELINE_EVAL=/root/autodl-tmp/Runs/EXP-0022__minecraft-runtime-milestones__s10000__3eps__20260812T200000Z/full-36000/evaluation/evaluation.json
readonly INVENTORY=/root/autodl-tmp/Runs/EXP-0012__minecraft-diamond__s31415__l0-32-step__20260812T080000Z/l0/minecraft_l0.json
readonly OUTPUT=/root/autodl-tmp/Artifacts/dreamerv3/review/EXP-0023-minecraft-exact-500k

if [[ -e "${OUTPUT}" ]]; then
  mapfile -t existing < <(find "${OUTPUT}" -mindepth 1 -maxdepth 1 -printf '%f\n')
  if (( ${#existing[@]} != 1 )) || [[ "${existing[0]}" != README.md ]] || \
      ! grep -q '尚未生成紧凑图' "${OUTPUT}/README.md"; then
    echo "Refusing to overwrite non-placeholder ${EXPERIMENT} review output" >&2
    exit 20
  fi
fi
"${PYTHON}" "${CONTROL}/scripts/analyze_exp0023_increment.py" \
  --experiment-id "${EXPERIMENT}" \
  --train-run "${TRAIN}" --eval-run "${EVAL}" \
  --source-replay "${SOURCE}/train/replay" \
  --baseline-evaluation "${BASELINE_EVAL}" --inventory-schema "${INVENTORY}" \
  --output-dir "${OUTPUT}"
sha256sum "${OUTPUT}"/* > "${OUTPUT}/SHA256SUMS"
