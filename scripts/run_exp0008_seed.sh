#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <seed: 0|1|2|3|4>" >&2
  exit 2
fi

readonly SEED=$1
case "${SEED}" in
  0|1|2|3|4) ;;
  *) echo "Seed must be one of 0, 1, 2, 3, 4" >&2; exit 2 ;;
esac

readonly EXPERIMENT=EXP-0008
readonly TAG=EXP-0008__cheetah-run__five-seed__500k-env__20260805T160000Z
readonly ROOT=/root/autodl-tmp/runs/${TAG}
readonly SIGNAL=/root/autodl-tmp/runs/${TAG}
readonly SEED_PADDED=$(printf '%03d' "${SEED}")
readonly OUTPUT=${ROOT}/s${SEED_PADDED}
readonly TRAIN_OUTPUT=${OUTPUT}/train
readonly RUNTIME=/root/autodl-tmp/dreamerv3-2411f7d
readonly CONTROL=/root/autodl-tmp/dreamerv3-reproduction
readonly PYTHON=/root/autodl-tmp/envs/dv3-2411/bin/python
readonly CONFIG=${CONTROL}/docs/reproduction/configs/exp0008_cheetah_s${SEED_PADDED}_500k_env.yaml
readonly VERIFY=${CONTROL}/scripts/verify_exp0008_run.py

if [[ ! -f "${SIGNAL}.freeze" ]]; then
  echo "Missing ${SIGNAL}.freeze" >&2
  exit 20
fi
if [[ ! -f "${CONFIG}" ]]; then
  echo "Missing frozen config: ${CONFIG}" >&2
  exit 21
fi
if [[ -e "${OUTPUT}" || -e "${OUTPUT}.started" ]]; then
  echo "Refusing duplicate launch: ${OUTPUT}" >&2
  exit 22
fi
mapfile -t gpu_pids < <(
  nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)
if (( ${#gpu_pids[@]} )); then
  echo "GPU is busy: ${gpu_pids[*]}" >&2
  exit 23
fi
available_kib=$(df --output=avail /root/autodl-tmp | tail -1)
if (( available_kib < 8 * 1024 * 1024 )); then
  echo "Data disk has less than the frozen 8 GiB floor" >&2
  exit 24
fi

find /dev/shm -maxdepth 1 -type f \
  \( -name 'torch_*' -o -name 'sem.loky-*' -o -name 'cuda.shm.*' \) \
  -delete
mkdir -p "$(dirname "${OUTPUT}")"
if ! (set -o noclobber; printf \
    '{"experiment_id":"%s","seed":%d,"started_at":"%s","pid":%d}\n' \
    "${EXPERIMENT}" "${SEED}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$$" \
    > "${OUTPUT}.started"); then
  echo "Refusing duplicate launch: ${OUTPUT}.started exists" >&2
  exit 25
fi
mkdir -p "${OUTPUT}"
cp "${SIGNAL}.freeze" "${OUTPUT}/.freeze"
cp "${OUTPUT}.started" "${OUTPUT}/.started"

fail() {
  local status=$1
  local phase=$2
  printf \
    '{"experiment_id":"%s","seed":%d,"failed_at":"%s","phase":"%s","exit_code":%d}\n' \
    "${EXPERIMENT}" "${SEED}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    "${phase}" "${status}" > "${OUTPUT}.failed"
  cp "${OUTPUT}.failed" "${OUTPUT}/.failed"
  exit "${status}"
}

cd "${RUNTIME}"
env PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
  "${PYTHON}" dreamerv3/main.py \
  --logdir "${TRAIN_OUTPUT}" \
  --configs dmc_proprio size12m \
  --task dmc_cheetah_run \
  --run.script train \
  --seed "${SEED}" \
  --tensorboard False \
  --env.dmc.repeat 2 \
  --run.num_envs 16 \
  --run.steps 250000 \
  --run.train_ratio 512 \
  --run.save_every 600 \
  --run.save_at_end True \
  > "${OUTPUT}/train_stdout.log" 2>&1 || fail $? train

"${PYTHON}" "${VERIFY}" \
  --run-dir "${OUTPUT}" \
  --frozen-config "${CONFIG}" \
  --expected-step 250000 \
  --output "${OUTPUT}/integrity.json" \
  > "${OUTPUT}/integrity_stdout.log" 2>&1 || fail $? integrity

cp "${OUTPUT}/integrity.json" "${OUTPUT}.integrity.json"
printf \
  '{"experiment_id":"%s","seed":%d,"completed_at":"%s","exit_code":0,"checkpoint_step":250000}\n' \
  "${EXPERIMENT}" "${SEED}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  > "${OUTPUT}.completed"
cp "${OUTPUT}.completed" "${OUTPUT}/.completed"
