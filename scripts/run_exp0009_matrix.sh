#!/usr/bin/env bash
set -euo pipefail

readonly EXPERIMENT=EXP-0009
readonly TAG=EXP-0009__reacher-hard__three-arm__s000__1m-env__20260806T034000Z
readonly ROOT=/root/autodl-tmp/Runs/${TAG}
readonly SIGNAL=/root/autodl-tmp/Runs/${TAG}
readonly CONTROL=/root/autodl-tmp/Code/DreamerV3/dreamerv3-reproduction
readonly RUN_ONE=${CONTROL}/scripts/run_exp0009_arm.sh
readonly ANALYZE=${CONTROL}/scripts/analyze_exp0009.py
readonly PYTHON=/root/autodl-tmp/Envs/dv3-2411/bin/python
readonly ARTIFACTS=/root/autodl-tmp/Artifacts/dreamerv3/EXP-0009

if [[ ! -f "${SIGNAL}.freeze" ]]; then
  echo "Missing ${SIGNAL}.freeze" >&2
  exit 20
fi
if [[ -e "${ROOT}" || -e "${SIGNAL}.started" ]]; then
  echo "Refusing duplicate matrix launch: ${TAG}" >&2
  exit 21
fi
mapfile -t gpu_pids < <(
  nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)
if (( ${#gpu_pids[@]} )); then
  echo "GPU is busy: ${gpu_pids[*]}" >&2
  exit 22
fi
available_kib=$(df --output=avail /root/autodl-tmp | tail -1)
if (( available_kib < 10 * 1024 * 1024 )); then
  echo "Data disk has less than the frozen 10 GiB launch floor" >&2
  exit 23
fi
if ! (set -o noclobber; printf \
    '{"experiment_id":"%s","started_at":"%s","pid":%d,"order":["baseline","no_reward_value","no_reconstruction"]}\n' \
    "${EXPERIMENT}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$$" \
    > "${SIGNAL}.started"); then
  echo "Refusing duplicate matrix launch: ${SIGNAL}.started exists" >&2
  exit 24
fi
mkdir -p "${ROOT}"
cp "${SIGNAL}.freeze" "${ROOT}/.freeze"
cp "${SIGNAL}.started" "${ROOT}/.started"
readonly START_EPOCH=$(date +%s)

fail() {
  local status=$1
  local phase=$2
  printf \
    '{"experiment_id":"%s","failed_at":"%s","phase":"%s","exit_code":%d,"wall_seconds":%d}\n' \
    "${EXPERIMENT}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${phase}" \
    "${status}" "$(( $(date +%s) - START_EPOCH ))" > "${SIGNAL}.failed"
  cp "${SIGNAL}.failed" "${ROOT}/.failed"
  exit "${status}"
}

for arm in baseline no_reward_value no_reconstruction; do
  "${RUN_ONE}" "${arm}" || fail $? "${arm}"
done

"${PYTHON}" "${ANALYZE}" --run-root "${ROOT}" --output-dir "${ARTIFACTS}" \
  > "${ROOT}/analysis_stdout.log" 2>&1 || fail $? analysis

printf \
  '{"experiment_id":"%s","completed_at":"%s","exit_code":0,"runs":3,"wall_seconds":%d,"analysis":"%s"}\n' \
  "${EXPERIMENT}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  "$(( $(date +%s) - START_EPOCH ))" "${ARTIFACTS}" > "${SIGNAL}.completed"
cp "${SIGNAL}.completed" "${ROOT}/.completed"
