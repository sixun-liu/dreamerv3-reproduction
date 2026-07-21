#!/usr/bin/env bash
set -euo pipefail

readonly EXPERIMENT=EXP-0006
readonly TAG=EXP-0006__walker-kl__matrix-2seed__20260721T142500Z
readonly ROOT=/root/autodl-tmp/runs/${TAG}
readonly SIGNAL=/root/autodl-tmp/runs/${TAG}
readonly CONTROL=/root/autodl-tmp/dreamerv3-reproduction
readonly RUN_ONE=${CONTROL}/scripts/run_exp0006_arm_seed.sh

if [[ ! -f "${SIGNAL}.freeze" ]]; then
  echo "Missing ${SIGNAL}.freeze" >&2
  exit 20
fi
if [[ -e "${ROOT}" || -e "${SIGNAL}.started" ]]; then
  echo "Refusing duplicate matrix launch: ${TAG}" >&2
  exit 21
fi
if ! (set -o noclobber; printf \
    '{"experiment_id":"%s","started_at":"%s","pid":%d,"order":["baseline:0","e1:0","p4:0","p4:1","e1:1","baseline:1"]}\n' \
    "${EXPERIMENT}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$$" \
    > "${SIGNAL}.started"); then
  echo "Refusing duplicate matrix launch: ${SIGNAL}.started exists" >&2
  exit 22
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

for item in baseline:0 e1:0 p4:0 p4:1 e1:1 baseline:1; do
  arm=${item%%:*}
  seed=${item##*:}
  "${RUN_ONE}" "${arm}" "${seed}" || fail $? "${arm}_s${seed}"
done

printf '{"experiment_id":"%s","completed_at":"%s","exit_code":0,"runs":6}\n' \
  "${EXPERIMENT}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  > "${SIGNAL}.completed"
cp "${SIGNAL}.completed" "${ROOT}/.completed"
