#!/usr/bin/env bash
set -uo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <seed: 0|1|2>" >&2
  exit 2
fi
readonly SEED=$1
case "${SEED}" in
  0) readonly TAG=EXP-0004__walker_walk__s000__500k-env__20260721T061500Z ;;
  1) readonly TAG=EXP-0004__walker_walk__s001__500k-env__20260721T061500Z ;;
  2) readonly TAG=EXP-0004__walker_walk__s002__500k-env__20260721T061500Z ;;
  *) echo "Seed must be one of 0, 1, 2" >&2; exit 2 ;;
esac

readonly OUTPUT=/root/autodl-tmp/runs/${TAG}
readonly TRAIN_OUTPUT=${OUTPUT}/train
readonly EVAL_OUTPUT=${OUTPUT}/eval
readonly RUNTIME=/root/autodl-tmp/dreamerv3
readonly PYTHON=/root/miniconda3/envs/dv3/bin/python

mkdir -p "${OUTPUT}"
if [[ ! -f "${OUTPUT}/.freeze" ]]; then
  echo "Missing ${OUTPUT}/.freeze" >&2
  exit 20
fi
if ! (set -o noclobber; printf '{"experiment_id":"EXP-0004","seed":%d,"started_at":"%s","pid":%d}\n' \
    "${SEED}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$$" > "${OUTPUT}/.started"); then
  echo "Refusing duplicate launch: ${OUTPUT}/.started exists" >&2
  exit 21
fi

fail() {
  local status=$1
  local phase=$2
  printf '{"experiment_id":"EXP-0004","seed":%d,"failed_at":"%s","phase":"%s","exit_code":%d}\n' \
    "${SEED}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${phase}" "${status}" > "${OUTPUT}/.failed"
  exit "${status}"
}

cd "${RUNTIME}"
env PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MUJOCO_GL=egl \
  "${PYTHON}" dreamerv3/main.py \
  --logdir "${TRAIN_OUTPUT}" \
  --configs dmc_proprio size12m \
  --task dmc_walker_walk \
  --script train \
  --seed "${SEED}" \
  --env.dmc.repeat 2 \
  --run.envs 16 \
  --run.steps 250000 \
  --run.train_ratio 512 \
  --run.save_every 600 \
  --run.save_at_end True \
  > "${OUTPUT}/train_stdout.log" 2>&1 || fail $? train
printf '{"seed":%d,"completed_at":"%s","exit_code":0}\n' \
  "${SEED}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${OUTPUT}/train.completed"

latest=$(<"${TRAIN_OUTPUT}/ckpt/latest")
checkpoint="${TRAIN_OUTPUT}/ckpt/${latest}"
printf '%s\n' "${checkpoint}" > "${OUTPUT}/checkpoint_path.txt"
checkpoint_step=$("${PYTHON}" -c \
  "import cloudpickle,pathlib; print(cloudpickle.loads(pathlib.Path('${checkpoint}/step.pkl').read_bytes()))") \
  || fail $? checkpoint_step
printf '%s\n' "${checkpoint_step}" > "${OUTPUT}/checkpoint_step.txt"
if [[ "${checkpoint_step}" != "250000" ]]; then
  echo "Expected final checkpoint step 250000, got ${checkpoint_step}" >&2
  fail 22 checkpoint_step
fi

env PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MUJOCO_GL=egl \
  "${PYTHON}" dreamerv3/main.py \
  --logdir "${EVAL_OUTPUT}" \
  --configs dmc_proprio size12m \
  --task dmc_walker_walk \
  --script eval_only \
  --seed 10000 \
  --env.dmc.repeat 2 \
  --run.envs 16 \
  --run.steps 40000 \
  --run.from_checkpoint "${checkpoint}" \
  > "${OUTPUT}/eval_stdout.log" 2>&1 || fail $? eval
printf '{"seed":%d,"completed_at":"%s","exit_code":0}\n' \
  "${SEED}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${OUTPUT}/eval.completed"

printf '{"experiment_id":"EXP-0004","seed":%d,"completed_at":"%s","exit_code":0,"checkpoint_step":250000}\n' \
  "${SEED}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${OUTPUT}/.completed"
