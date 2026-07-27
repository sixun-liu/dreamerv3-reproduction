#!/usr/bin/env bash
set -euo pipefail

readonly EXPERIMENT=EXP-0007
readonly TAG=EXP-0007__openloop-prediction__20260727T040500Z
readonly ROOT=/root/autodl-tmp/runs/${TAG}
readonly SIGNAL=/root/autodl-tmp/runs/${TAG}
readonly CONTROL=/root/autodl-tmp/dreamerv3-reproduction
readonly RUNTIME=/root/autodl-tmp/dreamerv3-exp0007
readonly MATRIX=/root/autodl-tmp/runs/EXP-0006__walker-kl__matrix-2seed__20260721T142500Z
readonly PANEL_ROOT=/root/autodl-tmp/staging/EXP-0007-panel-20260727T040500Z
readonly PANEL=${PANEL_ROOT}/panel.npz
readonly MANIFEST=${PANEL_ROOT}/panel_manifest.json
readonly EVALUATOR=${CONTROL}/scripts/evaluate_exp0007_openloop.py

if [[ ! -f "${SIGNAL}.freeze" ]]; then
  echo "Missing ${SIGNAL}.freeze" >&2
  exit 20
fi
if [[ -e "${ROOT}" || -e "${SIGNAL}.started" ]]; then
  echo "Refusing duplicate probe launch: ${TAG}" >&2
  exit 21
fi
if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; then
  echo "GPU already has a compute process" >&2
  exit 22
fi
available_kib=$(df -Pk /root/autodl-tmp | awk 'NR==2 {print $4}')
if (( available_kib < 10 * 1024 * 1024 )); then
  echo "Less than 10 GiB free on data disk" >&2
  exit 23
fi
python "${CONTROL}/scripts/openloop_panel.py" verify \
  --panel "${PANEL}" --manifest "${MANIFEST}" --source --hashes

if ! (set -o noclobber; printf \
    '{"experiment_id":"%s","started_at":"%s","pid":%d,"runs":6}\n' \
    "${EXPERIMENT}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$$" \
    > "${SIGNAL}.started"); then
  echo "Refusing duplicate probe launch: ${SIGNAL}.started exists" >&2
  exit 24
fi
mkdir -p "${ROOT}"
cp "${SIGNAL}.freeze" "${ROOT}/.freeze"
cp "${SIGNAL}.started" "${ROOT}/.started"

fail() {
  local status=$1
  local phase=$2
  printf \
    '{"experiment_id":"%s","failed_at":"%s","phase":"%s","exit_code":%d}\n' \
    "${EXPERIMENT}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    "${phase}" "${status}" > "${SIGNAL}.failed"
  cp "${SIGNAL}.failed" "${ROOT}/.failed"
  exit "${status}"
}

for item in baseline:0 baseline:1 e1:0 e1:1 p4:0 p4:1; do
  arm=${item%%:*}
  seed=${item##*:}
  run=${MATRIX}/${arm}/s$(printf '%03d' "${seed}")
  output=${ROOT}/${arm}/s$(printf '%03d' "${seed}")
  mkdir -p "${output}"
  printf \
    '{"experiment_id":"%s","arm":"%s","seed":%d,"started_at":"%s"}\n' \
    "${EXPERIMENT}" "${arm}" "${seed}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    > "${output}/.started"
  PYOPENGL_PLATFORM=egl MUJOCO_GL=egl conda run -n dv3 python "${EVALUATOR}" \
    --runtime "${RUNTIME}" \
    --panel "${PANEL}" \
    --manifest "${MANIFEST}" \
    --run-dir "${run}" \
    --arm "${arm}" \
    --seed "${seed}" \
    --output "${output}/predictions.npz" \
    --metadata "${output}/metadata.json" \
    --context 16 \
    --draws 4 \
    --batch-windows 16 \
    --eval-seed 20260727 \
    > "${output}/stdout.log" 2>&1 || fail $? "${arm}_s${seed}"
  printf \
    '{"experiment_id":"%s","arm":"%s","seed":%d,"completed_at":"%s","exit_code":0}\n' \
    "${EXPERIMENT}" "${arm}" "${seed}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    > "${output}/.completed"
done

printf '{"experiment_id":"%s","completed_at":"%s","exit_code":0,"runs":6}\n' \
  "${EXPERIMENT}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  > "${SIGNAL}.completed"
cp "${SIGNAL}.completed" "${ROOT}/.completed"
