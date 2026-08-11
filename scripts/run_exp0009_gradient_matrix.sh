#!/usr/bin/env bash
set -euo pipefail

readonly TAG=EXP-0009__reacher-hard__gradient-gate__s000__debug-120-dec__20260806T032742Z
readonly ROOT=/root/autodl-tmp/Runs/${TAG}
readonly CONTROL=/root/autodl-tmp/Code/DreamerV3/dreamerv3-reproduction
readonly RUNTIME=/root/autodl-tmp/Code/DreamerV3/dreamerv3-2411f7d
readonly PYTHON=/root/autodl-tmp/Envs/dv3-2411/bin/python
readonly RUN_ONE=${CONTROL}/scripts/run_exp0009_gradient_probe.sh
readonly VERIFY=${CONTROL}/scripts/verify_exp0009_gradient_gate.py

if [[ -e "${ROOT}" || -e "${ROOT}.started" ]]; then
  echo "Refusing duplicate gradient matrix: ${ROOT}" >&2
  exit 20
fi
mapfile -t gpu_pids < <(
  nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)
if (( ${#gpu_pids[@]} )); then
  echo "GPU is busy: ${gpu_pids[*]}" >&2
  exit 21
fi
if [[ -n "$(git -C "${CONTROL}" status --porcelain)" ]] || \
   [[ -n "$(git -C "${RUNTIME}" status --porcelain)" ]]; then
  echo "Control and runtime worktrees must be clean" >&2
  exit 22
fi

mkdir -p "${ROOT}"
printf \
  '{"experiment_id":"EXP-0009","started_at":"%s","pid":%d,"scope":"gradient_gate","order":["baseline","no_reward_value","no_reconstruction"]}\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$$" > "${ROOT}.started"
cp "${ROOT}.started" "${ROOT}/.started"
printf \
  '{"experiment_id":"EXP-0009","scope":"gradient_gate","runtime_commit":"%s","control_commit":"%s","seed":0,"agent_decisions":120,"environment_steps":240}\n' \
  "$(git -C "${RUNTIME}" rev-parse HEAD)" "$(git -C "${CONTROL}" rev-parse HEAD)" \
  > "${ROOT}/.freeze"

fail() {
  local status=$1
  local phase=$2
  printf \
    '{"experiment_id":"EXP-0009","failed_at":"%s","phase":"%s","exit_code":%d,"scope":"gradient_gate"}\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${phase}" "${status}" > "${ROOT}.failed"
  cp "${ROOT}.failed" "${ROOT}/.failed"
  exit "${status}"
}

(cd "${RUNTIME}" && "${PYTHON}" -m unittest -v \
  dreamerv3.tests.test_gradient_routing) \
  > "${ROOT}/algebraic_test.log" 2>&1 || fail $? algebraic

for arm in baseline no_reward_value no_reconstruction; do
  "${RUN_ONE}" "${arm}" || fail $? "${arm}"
done

"${PYTHON}" "${VERIFY}" --root "${ROOT}" --output "${ROOT}/gradient_gate.json" \
  > "${ROOT}/gradient_gate_stdout.log" 2>&1 || fail $? gradient_verification

printf \
  '{"experiment_id":"EXP-0009","completed_at":"%s","exit_code":0,"scope":"gradient_gate","passed":true}\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${ROOT}.completed"
cp "${ROOT}.completed" "${ROOT}/.completed"
