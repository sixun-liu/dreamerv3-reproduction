#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <baseline|no_reward_value|no_reconstruction>" >&2
  exit 2
fi

readonly ARM=$1
case "${ARM}" in
  baseline)
    readonly REWARD_GRAD=True
    readonly VALUE_GRAD=True
    readonly RECONSTRUCTION_GRAD=True
    ;;
  no_reward_value)
    readonly REWARD_GRAD=False
    readonly VALUE_GRAD=False
    readonly RECONSTRUCTION_GRAD=True
    ;;
  no_reconstruction)
    readonly REWARD_GRAD=True
    readonly VALUE_GRAD=True
    readonly RECONSTRUCTION_GRAD=False
    ;;
  *) echo "Unknown arm: ${ARM}" >&2; exit 2 ;;
esac

readonly EXPERIMENT=EXP-0009
readonly TAG=EXP-0009__reacher-hard__three-arm__s000__1m-env__20260806T034000Z
readonly ROOT=/root/autodl-tmp/runs/${TAG}
readonly SIGNAL=/root/autodl-tmp/runs/${TAG}
readonly OUTPUT=${ROOT}/${ARM}
readonly TRAIN_OUTPUT=${OUTPUT}/train
readonly RUNTIME=/root/autodl-tmp/dreamerv3-2411f7d
readonly CONTROL=/root/autodl-tmp/dreamerv3-reproduction
readonly PYTHON=/root/autodl-tmp/envs/dv3-2411/bin/python
readonly CONFIG=${CONTROL}/docs/reproduction/configs/exp0009_reacher_${ARM}_s000_1m_env.yaml
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
if (( available_kib < 5 * 1024 * 1024 )); then
  echo "Data disk has less than the frozen 5 GiB stop floor" >&2
  exit 24
fi

find /dev/shm -maxdepth 1 -type f \
  \( -name 'torch_*' -o -name 'sem.loky-*' -o -name 'cuda.shm.*' \) \
  -delete
mkdir -p "$(dirname "${OUTPUT}")"
if ! (set -o noclobber; printf \
    '{"experiment_id":"%s","arm":"%s","seed":0,"started_at":"%s","pid":%d}\n' \
    "${EXPERIMENT}" "${ARM}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$$" \
    > "${OUTPUT}.started"); then
  echo "Refusing duplicate launch: ${OUTPUT}.started exists" >&2
  exit 25
fi
mkdir -p "${OUTPUT}"
cp "${SIGNAL}.freeze" "${OUTPUT}/.freeze"
cp "${OUTPUT}.started" "${OUTPUT}/.started"
readonly START_EPOCH=$(date +%s)

{
  date -u +started_at=%Y-%m-%dT%H:%M:%SZ
  df -B1 /root/autodl-tmp
  nvidia-smi --query-gpu=name,memory.total,memory.free,utilization.gpu \
    --format=csv,noheader,nounits
} > "${OUTPUT}/resource_before.txt"

fail() {
  local status=$1
  local phase=$2
  printf \
    '{"experiment_id":"%s","arm":"%s","failed_at":"%s","phase":"%s","exit_code":%d,"wall_seconds":%d}\n' \
    "${EXPERIMENT}" "${ARM}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    "${phase}" "${status}" "$(( $(date +%s) - START_EPOCH ))" > "${OUTPUT}.failed"
  cp "${OUTPUT}.failed" "${OUTPUT}/.failed"
  exit "${status}"
}

cd "${RUNTIME}"
/usr/bin/time -v -o "${OUTPUT}/resource_time.txt" \
  env PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
  "${PYTHON}" dreamerv3/main.py \
  --logdir "${TRAIN_OUTPUT}" \
  --configs dmc_proprio size12m \
  --task dmc_reacher_hard \
  --run.script train \
  --seed 0 \
  --tensorboard False \
  --env.dmc.repeat 2 \
  --env.dmc.use_seed True \
  --run.num_envs 16 \
  --run.steps 500000 \
  --run.train_ratio 512 \
  --run.save_every 600 \
  --run.save_at_end True \
  --report_gradnorms False \
  --reward_grad "${REWARD_GRAD}" \
  --replay_critic_grad "${VALUE_GRAD}" \
  --reconstruction_grad "${RECONSTRUCTION_GRAD}" \
  > "${OUTPUT}/train_stdout.log" 2>&1 || fail $? train

{
  date -u +completed_training_at=%Y-%m-%dT%H:%M:%SZ
  df -B1 /root/autodl-tmp
  du -sb "${OUTPUT}"
  nvidia-smi --query-gpu=name,memory.total,memory.free,utilization.gpu \
    --format=csv,noheader,nounits
} > "${OUTPUT}/resource_after.txt"

"${PYTHON}" "${VERIFY}" \
  --run-dir "${OUTPUT}" \
  --frozen-config "${CONFIG}" \
  --expected-step 500000 \
  --output "${OUTPUT}/integrity.json" \
  > "${OUTPUT}/integrity_stdout.log" 2>&1 || fail $? integrity

readonly WALL_SECONDS=$(( $(date +%s) - START_EPOCH ))
readonly OUTPUT_BYTES=$(du -sb "${OUTPUT}" | awk '{print $1}')
cp "${OUTPUT}/integrity.json" "${OUTPUT}.integrity.json"
printf \
  '{"experiment_id":"%s","arm":"%s","seed":0,"completed_at":"%s","exit_code":0,"checkpoint_step":500000,"environment_steps":1000000,"wall_seconds":%d,"output_bytes":%d}\n' \
  "${EXPERIMENT}" "${ARM}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  "${WALL_SECONDS}" "${OUTPUT_BYTES}" > "${OUTPUT}.completed"
cp "${OUTPUT}.completed" "${OUTPUT}/.completed"
