#!/usr/bin/env bash
set -euo pipefail

readonly EXPERIMENT=EXP-0008
readonly TAG=EXP-0008__cheetah-run__five-seed__500k-env__20260805T160000Z
readonly ROOT=/root/autodl-tmp/runs/${TAG}
readonly SIGNAL=/root/autodl-tmp/runs/${TAG}
readonly CONTROL=/root/autodl-tmp/dreamerv3-reproduction
readonly RUN_ONE=${CONTROL}/scripts/run_exp0008_seed.sh

if [[ ! -f "${SIGNAL}.freeze" ]]; then
  echo "Missing ${SIGNAL}.freeze" >&2
  exit 20
fi
if [[ -e "${ROOT}" || -e "${SIGNAL}.started" ]]; then
  echo "Refusing duplicate matrix launch: ${TAG}" >&2
  exit 21
fi
if ! (set -o noclobber; printf \
    '{"experiment_id":"%s","started_at":"%s","pid":%d,"order":[0,1,2,3,4]}\n' \
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

for seed in 0 1 2 3 4; do
  "${RUN_ONE}" "${seed}" || fail $? "seed_${seed}"
done

printf '{"experiment_id":"%s","completed_at":"%s","exit_code":0,"runs":5}\n' \
  "${EXPERIMENT}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  > "${SIGNAL}.completed"
cp "${SIGNAL}.completed" "${ROOT}/.completed"
