#!/usr/bin/env bash
set -uo pipefail

readonly OUTPUT=/root/autodl-tmp/runs/EXP-0003__walker_walk__s31415__final-ckpt-smoke__20260721T055700Z
readonly TRAIN_OUTPUT=${OUTPUT}/train
readonly EVAL_OUTPUT=${OUTPUT}/eval
readonly RUNTIME=/root/autodl-tmp/dreamerv3
readonly PYTHON=/root/miniconda3/envs/dv3/bin/python

mkdir -p "${OUTPUT}"
if [[ ! -f "${OUTPUT}/.freeze" ]]; then
  echo "Missing ${OUTPUT}/.freeze" >&2
  exit 20
fi
if ! (set -o noclobber; printf '{"experiment_id":"EXP-0003","started_at":"%s","pid":%d}\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$$" > "${OUTPUT}/.started"); then
  echo "Refusing duplicate launch: ${OUTPUT}/.started exists" >&2
  exit 21
fi

fail() {
  local status=$1
  local phase=$2
  printf '{"experiment_id":"EXP-0003","failed_at":"%s","phase":"%s","exit_code":%d}\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${phase}" "${status}" > "${OUTPUT}/.failed"
  exit "${status}"
}

cd "${RUNTIME}"
env PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MUJOCO_GL=egl \
  "${PYTHON}" dreamerv3/main.py \
  --logdir "${TRAIN_OUTPUT}" \
  --configs dmc_proprio size12m \
  --task dmc_walker_walk \
  --script train \
  --seed 31415 \
  --env.dmc.repeat 2 \
  --run.envs 16 \
  --run.steps 2048 \
  --run.train_ratio 512 \
  --run.save_every 3600 \
  --run.save_at_end True \
  > "${OUTPUT}/train_stdout.log" 2>&1 || fail $? train
printf '{"completed_at":"%s","exit_code":0}\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${OUTPUT}/train.completed"

latest=$(<"${TRAIN_OUTPUT}/ckpt/latest")
checkpoint="${TRAIN_OUTPUT}/ckpt/${latest}"
printf '%s\n' "${checkpoint}" > "${OUTPUT}/checkpoint_path.txt"
checkpoint_step=$("${PYTHON}" -c \
  "import cloudpickle,pathlib; print(cloudpickle.loads(pathlib.Path('${checkpoint}/step.pkl').read_bytes()))") \
  || fail $? checkpoint_step
printf '%s\n' "${checkpoint_step}" > "${OUTPUT}/checkpoint_step.txt"
if [[ "${checkpoint_step}" != "2048" ]]; then
  echo "Expected final checkpoint step 2048, got ${checkpoint_step}" >&2
  fail 22 checkpoint_step
fi

env PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MUJOCO_GL=egl \
  "${PYTHON}" dreamerv3/main.py \
  --logdir "${EVAL_OUTPUT}" \
  --configs dmc_proprio size12m \
  --task dmc_walker_walk \
  --script eval_only \
  --seed 131415 \
  --env.dmc.repeat 2 \
  --run.envs 16 \
  --run.steps 8320 \
  --run.from_checkpoint "${checkpoint}" \
  > "${OUTPUT}/eval_stdout.log" 2>&1 || fail $? eval
printf '{"completed_at":"%s","exit_code":0}\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${OUTPUT}/eval.completed"

printf '{"experiment_id":"EXP-0003","completed_at":"%s","exit_code":0,"checkpoint_step":2048}\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${OUTPUT}/.completed"
