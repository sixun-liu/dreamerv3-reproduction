#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <baseline|e1|p4> <seed: 0|1>" >&2
  exit 2
fi

readonly ARM=$1
readonly SEED=$2
case "${ARM}" in
  baseline|e1|p4) ;;
  *) echo "Unknown arm: ${ARM}" >&2; exit 2 ;;
esac
case "${SEED}" in
  0|1) ;;
  *) echo "Seed must be 0 or 1" >&2; exit 2 ;;
esac

readonly EXPERIMENT=EXP-0006
readonly TAG=EXP-0006__walker-kl__matrix-2seed__20260721T142500Z
readonly ROOT=/root/autodl-tmp/runs/${TAG}
readonly SIGNAL=/root/autodl-tmp/runs/${TAG}
readonly OUTPUT=${ROOT}/${ARM}/s$(printf '%03d' "${SEED}")
readonly TRAIN_OUTPUT=${OUTPUT}/train
readonly EVAL_OUTPUT=${OUTPUT}/eval
readonly RUNTIME=/root/autodl-tmp/dreamerv3-exp0006
readonly CONTROL=/root/autodl-tmp/dreamerv3-reproduction
readonly PYTHON=/root/miniconda3/envs/dv3/bin/python
readonly TRAIN_CONFIG=${CONTROL}/docs/reproduction/configs/exp0006_${ARM}_s$(printf '%03d' "${SEED}")_train.yaml
readonly EVAL_CONFIG=${CONTROL}/docs/reproduction/configs/exp0006_${ARM}_s$(printf '%03d' "${SEED}")_eval.yaml

if [[ ! -f "${SIGNAL}.freeze" ]]; then
  echo "Missing ${SIGNAL}.freeze" >&2
  exit 20
fi
if [[ ! -f "${TRAIN_CONFIG}" || ! -f "${EVAL_CONFIG}" ]]; then
  echo "Missing frozen config for ${ARM} seed ${SEED}" >&2
  exit 21
fi
if [[ -e "${OUTPUT}" ]]; then
  echo "Refusing duplicate launch: ${OUTPUT} exists" >&2
  exit 22
fi
mapfile -t gpu_pids < <(
  nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits |
    sed '/^[[:space:]]*$/d')
if (( ${#gpu_pids[@]} )); then
  echo "GPU is busy: ${gpu_pids[*]}" >&2
  exit 23
fi
available_kib=$(df --output=avail /root/autodl-tmp | tail -1)
if (( available_kib < 10 * 1024 * 1024 )); then
  echo "Data disk has less than 10 GiB available" >&2
  exit 24
fi

find /dev/shm -maxdepth 1 -type f \
  \( -name 'torch_*' -o -name 'sem.loky-*' -o -name 'cuda.shm.*' \) \
  -delete
mkdir -p "$(dirname "${OUTPUT}")"
if ! (set -o noclobber; printf \
    '{"experiment_id":"%s","arm":"%s","seed":%d,"started_at":"%s","pid":%d}\n' \
    "${EXPERIMENT}" "${ARM}" "${SEED}" \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$$" > "${OUTPUT}.started"); then
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
    '{"experiment_id":"%s","arm":"%s","seed":%d,"failed_at":"%s","phase":"%s","exit_code":%d}\n' \
    "${EXPERIMENT}" "${ARM}" "${SEED}" \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${phase}" "${status}" \
    > "${OUTPUT}.failed"
  cp "${OUTPUT}.failed" "${OUTPUT}/.failed"
  exit "${status}"
}

arm_overrides=()
case "${ARM}" in
  baseline) ;;
  e1)
    arm_overrides+=(--agent.dyn.rssm.free_nats 0.0)
    ;;
  p4)
    arm_overrides+=(
      --agent.dyn.rssm.free_nats 0.0
      --agent.loss_scales.rep 1.0)
    ;;
esac

cd "${RUNTIME}"
env PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
  "${PYTHON}" dreamerv3/main.py \
  --logdir "${TRAIN_OUTPUT}" \
  --configs dmc_proprio size12m \
  --task dmc_walker_walk \
  --script train \
  --seed "${SEED}" \
  --env.dmc.use_seed True \
  --env.dmc.repeat 2 \
  --run.envs 16 \
  --run.steps 250000 \
  --run.train_ratio 512 \
  --run.log_every 120 \
  --run.report_every 300 \
  --run.save_every 600 \
  --run.save_at_end True \
  "${arm_overrides[@]}" \
  > "${OUTPUT}/train_stdout.log" 2>&1 || fail $? train

printf '{"arm":"%s","seed":%d,"completed_at":"%s","exit_code":0}\n' \
  "${ARM}" "${SEED}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  > "${OUTPUT}/train.completed"

latest=$(<"${TRAIN_OUTPUT}/ckpt/latest")
readonly CHECKPOINT=${TRAIN_OUTPUT}/ckpt/${latest}
if [[ ! -s "${CHECKPOINT}/agent.pkl" || ! -s "${CHECKPOINT}/step.pkl" ]]; then
  echo "Missing terminal checkpoint: ${CHECKPOINT}" >&2
  fail 26 checkpoint
fi
checkpoint_step=$("${PYTHON}" -c \
  "import cloudpickle,pathlib; print(cloudpickle.loads(pathlib.Path('${CHECKPOINT}/step.pkl').read_bytes()))") \
  || fail $? checkpoint_step
printf '%s\n' "${CHECKPOINT}" > "${OUTPUT}/checkpoint_path.txt"
printf '%s\n' "${checkpoint_step}" > "${OUTPUT}/checkpoint_step.txt"
if [[ "${checkpoint_step}" != "250000" ]]; then
  echo "Expected checkpoint step 250000, got ${checkpoint_step}" >&2
  fail 27 checkpoint_step
fi
sha256sum "${CHECKPOINT}/agent.pkl" "${CHECKPOINT}/step.pkl" \
  > "${OUTPUT}/checkpoint.sha256"

env PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
  "${PYTHON}" dreamerv3/main.py \
  --logdir "${EVAL_OUTPUT}" \
  --configs dmc_proprio size12m \
  --task dmc_walker_walk \
  --script eval_only \
  --seed 10000 \
  --env.dmc.use_seed True \
  --env.dmc.repeat 2 \
  --run.envs 16 \
  --run.steps 40000 \
  --run.from_checkpoint "${CHECKPOINT}" \
  "${arm_overrides[@]}" \
  > "${OUTPUT}/eval_stdout.log" 2>&1 || fail $? eval

printf '{"arm":"%s","seed":%d,"completed_at":"%s","exit_code":0}\n' \
  "${ARM}" "${SEED}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  > "${OUTPUT}/eval.completed"
printf \
  '{"experiment_id":"%s","arm":"%s","seed":%d,"completed_at":"%s","exit_code":0,"checkpoint_step":250000}\n' \
  "${EXPERIMENT}" "${ARM}" "${SEED}" \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${OUTPUT}.completed"
cp "${OUTPUT}.completed" "${OUTPUT}/.completed"
