#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <unique-eval-tag>" >&2
  exit 2
fi

readonly TAG=$1
if [[ ! "${TAG}" =~ ^EXP-0010__walker-vision__eval-s10000-10eps__[0-9]{8}T[0-9]{6}Z$ ]]; then
  echo "Invalid eval tag: ${TAG}" >&2
  exit 2
fi

readonly EXPERIMENT=EXP-0010
readonly CONTROL=/root/autodl-tmp/Code/DreamerV3/dreamerv3-reproduction
readonly RUNTIME=/root/autodl-tmp/Code/DreamerV3/dreamerv3-runtime-2411-crossdomain
readonly PYTHON=/root/autodl-tmp/Envs/dv3-2411/bin/python
readonly TRAIN_ROOT=/root/autodl-tmp/Runs/EXP-0010__walker-vision__s000__staged-1m-env__20260811T151208Z
readonly CHECKPOINT=${TRAIN_ROOT}/train/checkpoint.ckpt
readonly EVALUATE=${CONTROL}/scripts/evaluate_exp0010_checkpoint.py
readonly ROOT=/root/autodl-tmp/Runs/${TAG}

if [[ -e "${ROOT}" || -e "${ROOT}.started" ]]; then
  echo "Refusing duplicate evaluation launch" >&2
  exit 20
fi
if [[ ! -s "${CHECKPOINT}" || ! -f "${TRAIN_ROOT}/.full.completed" ]]; then
  echo "Final EXP-0010 checkpoint is incomplete" >&2
  exit 21
fi
mapfile -t gpu_pids < <(
  nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits |
    sed '/^[[:space:]]*$/d')
if (( ${#gpu_pids[@]} )); then
  echo "GPU is busy: ${gpu_pids[*]}" >&2
  exit 22
fi

mkdir -p "${ROOT}"
control_commit=$(git -C "${CONTROL}" rev-parse HEAD)
runtime_commit=$(git -C "${RUNTIME}" rev-parse HEAD)
checkpoint_sha=$(sha256sum "${CHECKPOINT}" | cut -d' ' -f1)
printf \
  '{"experiment_id":"%s","purpose":"independent terminal-checkpoint evaluation","created_at":"%s","control_commit":"%s","runtime_commit":"%s","checkpoint":"%s","checkpoint_sha256":"%s","agent_seed":10000,"environment_seed_controlled":false,"episodes":10,"video_selection":"episode index 0"}\n' \
  "${EXPERIMENT}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${control_commit}" \
  "${runtime_commit}" "${CHECKPOINT}" "${checkpoint_sha}" > "${ROOT}.freeze"
cp "${ROOT}.freeze" "${ROOT}/.freeze"
printf \
  '{"experiment_id":"%s","tag":"%s","started_at":"%s","pid":%d}\n' \
  "${EXPERIMENT}" "${TAG}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$$" \
  > "${ROOT}.started"
cp "${ROOT}.started" "${ROOT}/.started"

fail() {
  local status=$1
  printf \
    '{"experiment_id":"%s","tag":"%s","failed_at":"%s","exit_code":%d}\n' \
    "${EXPERIMENT}" "${TAG}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    "${status}" > "${ROOT}.failed"
  cp "${ROOT}.failed" "${ROOT}/.failed"
  exit "${status}"
}

cd "${CONTROL}"
/usr/bin/time -v -o "${ROOT}/resource_time.txt" \
  env PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
  "${PYTHON}" "${EVALUATE}" \
  --runtime "${RUNTIME}" \
  --checkpoint "${CHECKPOINT}" \
  --output "${ROOT}/evaluation" \
  --episodes 10 \
  --agent-seed 10000 \
  > "${ROOT}/stdout.log" 2>&1 || fail $?

if [[ ! -s "${ROOT}/evaluation/evaluation.json" || \
      ! -s "${ROOT}/evaluation/episode_000_preregistered.mp4" ]]; then
  fail 23
fi
sha256sum "${ROOT}/evaluation/evaluation.json" \
  "${ROOT}/evaluation/episode_000_preregistered.mp4" > "${ROOT}/SHA256SUMS"
printf \
  '{"experiment_id":"%s","tag":"%s","completed_at":"%s","exit_code":0,"episodes":10}\n' \
  "${EXPERIMENT}" "${TAG}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  > "${ROOT}.completed"
cp "${ROOT}.completed" "${ROOT}/.completed"
