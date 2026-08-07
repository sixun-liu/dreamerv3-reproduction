#!/usr/bin/env bash
set -euo pipefail

readonly EXPERIMENT=EXP-0008
readonly TAG=EXP-0008__cheetah-run__s31415__smoke-16384-dec__20260805T163000Z
readonly OUTPUT=/root/autodl-tmp/Runs/${TAG}
readonly SIGNAL=/root/autodl-tmp/Runs/${TAG}
readonly RUNTIME=/root/autodl-tmp/Code/DreamerV3/dreamerv3-2411f7d
readonly CONTROL=/root/autodl-tmp/Code/DreamerV3/dreamerv3-reproduction
readonly PYTHON=/root/autodl-tmp/Envs/dv3-2411/bin/python
readonly CONFIG=${CONTROL}/docs/reproduction/configs/exp0008_cheetah_smoke_s31415_16384_dec.yaml
readonly VERIFY=${CONTROL}/scripts/verify_exp0008_run.py

if [[ ! -f "${SIGNAL}.freeze" ]]; then
  echo "Missing ${SIGNAL}.freeze" >&2
  exit 20
fi
if [[ -e "${OUTPUT}" || -e "${SIGNAL}.started" ]]; then
  echo "Refusing duplicate smoke launch: ${OUTPUT}" >&2
  exit 21
fi
mapfile -t gpu_pids < <(
  nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)
if (( ${#gpu_pids[@]} )); then
  echo "GPU is busy: ${gpu_pids[*]}" >&2
  exit 22
fi
find /dev/shm -maxdepth 1 -type f \
  \( -name 'torch_*' -o -name 'sem.loky-*' -o -name 'cuda.shm.*' \) \
  -delete
if ! (set -o noclobber; printf \
    '{"experiment_id":"%s","started_at":"%s","pid":%d,"scope":"smoke"}\n' \
    "${EXPERIMENT}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$$" \
    > "${SIGNAL}.started"); then
  echo "Refusing duplicate smoke launch" >&2
  exit 23
fi
mkdir -p "${OUTPUT}"
cp "${SIGNAL}.freeze" "${OUTPUT}/.freeze"
cp "${SIGNAL}.started" "${OUTPUT}/.started"

fail() {
  local status=$1
  local phase=$2
  printf \
    '{"experiment_id":"%s","failed_at":"%s","phase":"%s","exit_code":%d,"scope":"smoke"}\n' \
    "${EXPERIMENT}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    "${phase}" "${status}" > "${SIGNAL}.failed"
  cp "${SIGNAL}.failed" "${OUTPUT}/.failed"
  exit "${status}"
}

cd "${RUNTIME}"
env PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
  "${PYTHON}" dreamerv3/main.py \
  --logdir "${OUTPUT}/train" \
  --configs dmc_proprio size12m \
  --task dmc_cheetah_run \
  --run.script train \
  --seed 31415 \
  --tensorboard False \
  --env.dmc.repeat 2 \
  --run.num_envs 16 \
  --run.steps 16384 \
  --run.train_ratio 512 \
  --run.log_every 30 \
  --run.save_every 60 \
  --run.save_at_end True \
  > "${OUTPUT}/train_stdout.log" 2>&1 || fail $? train

"${PYTHON}" "${VERIFY}" \
  --run-dir "${OUTPUT}" \
  --frozen-config "${CONFIG}" \
  --expected-step 16384 \
  --output "${OUTPUT}/integrity.json" \
  > "${OUTPUT}/integrity_stdout.log" 2>&1 || fail $? integrity

printf \
  '{"experiment_id":"%s","completed_at":"%s","exit_code":0,"checkpoint_step":16384,"scope":"smoke"}\n' \
  "${EXPERIMENT}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  > "${SIGNAL}.completed"
cp "${SIGNAL}.completed" "${OUTPUT}/.completed"
