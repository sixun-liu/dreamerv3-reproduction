#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <baseline|no_reward_value|no_reconstruction>" >&2
  exit 2
fi

readonly ARM=$1
case "${ARM}" in
  baseline) ;;
  no_reward_value) ;;
  no_reconstruction) ;;
  *) echo "Unknown arm: ${ARM}" >&2; exit 2 ;;
esac

readonly TAG=EXP-0009__reacher-hard__gradient-gate__s000__debug-120-dec__20260806T032742Z
readonly ROOT=/root/autodl-tmp/runs/${TAG}
readonly OUTPUT=${ROOT}/${ARM}
readonly RUNTIME=/root/autodl-tmp/dreamerv3-2411f7d
readonly CONTROL=/root/autodl-tmp/dreamerv3-reproduction
readonly PYTHON=/root/autodl-tmp/envs/dv3-2411/bin/python
readonly CONFIG=${CONTROL}/docs/reproduction/configs/exp0009_gradient_${ARM}_s000_120_dec.yaml
readonly VERIFY=${CONTROL}/scripts/verify_exp0009_gradient_probe.py

if [[ ! -f "${ROOT}/.freeze" ]]; then
  echo "Missing gradient-gate freeze: ${ROOT}/.freeze" >&2
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

find /dev/shm -maxdepth 1 -type f \
  \( -name 'torch_*' -o -name 'sem.loky-*' -o -name 'cuda.shm.*' \) \
  -delete
if ! (set -o noclobber; printf \
    '{"experiment_id":"EXP-0009","arm":"%s","started_at":"%s","pid":%d,"scope":"gradient_gate"}\n' \
    "${ARM}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$$" > "${OUTPUT}.started"); then
  echo "Refusing duplicate launch: ${OUTPUT}.started exists" >&2
  exit 24
fi
mkdir -p "${OUTPUT}"
cp "${ROOT}/.freeze" "${OUTPUT}/.freeze"
cp "${OUTPUT}.started" "${OUTPUT}/.started"

fail() {
  local status=$1
  local phase=$2
  printf \
    '{"experiment_id":"EXP-0009","arm":"%s","failed_at":"%s","phase":"%s","exit_code":%d,"scope":"gradient_gate"}\n' \
    "${ARM}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${phase}" "${status}" \
    > "${OUTPUT}.failed"
  cp "${OUTPUT}.failed" "${OUTPUT}/.failed"
  exit "${status}"
}

cd "${RUNTIME}"
env PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
  "${PYTHON}" dreamerv3/main.py \
  --logdir "${OUTPUT}/train" \
  --configs dmc_proprio debug \
  --task dmc_reacher_hard \
  --run.script train \
  --seed 0 \
  --tensorboard False \
  --jax.platform gpu \
  --jax.prealloc False \
  --env.dmc.repeat 2 \
  --env.dmc.use_seed True \
  --run.num_envs 4 \
  --run.steps 120 \
  --run.train_ratio 8 \
  --run.eval_every 0 \
  --run.log_every 0 \
  --run.save_every 1000000000 \
  --run.save_at_end True \
  --report_gradnorms True \
  --report_gradnorm_keys '^(reward|replay_critic|reconstruction)$' \
  --reward_grad "$([[ "${ARM}" == no_reward_value ]] && echo False || echo True)" \
  --replay_critic_grad "$([[ "${ARM}" == no_reward_value ]] && echo False || echo True)" \
  --reconstruction_grad "$([[ "${ARM}" == no_reconstruction ]] && echo False || echo True)" \
  > "${OUTPUT}/train_stdout.log" 2>&1 || fail $? train

"${PYTHON}" "${VERIFY}" \
  --run-dir "${OUTPUT}" \
  --frozen-config "${CONFIG}" \
  --expected-step 120 \
  --output "${OUTPUT}/integrity.json" \
  > "${OUTPUT}/integrity_stdout.log" 2>&1 || fail $? integrity

printf \
  '{"experiment_id":"EXP-0009","arm":"%s","completed_at":"%s","exit_code":0,"checkpoint_step":120,"scope":"gradient_gate"}\n' \
  "${ARM}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${OUTPUT}.completed"
cp "${OUTPUT}.completed" "${OUTPUT}/.completed"
