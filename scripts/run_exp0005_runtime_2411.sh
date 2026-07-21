#!/usr/bin/env bash
set -uo pipefail

readonly TAG=EXP-0005__walker_walk__s000-agent__500k-env__20260721T120000Z
readonly OUTPUT=/root/autodl-tmp/runs/${TAG}
readonly SIGNAL=/root/autodl-tmp/runs/${TAG}
readonly RUNTIME=/root/autodl-tmp/dreamerv3-2411f7d
readonly PYTHON=/root/autodl-tmp/envs/dv3-2411/bin/python

if [[ ! -f "${SIGNAL}.freeze" ]]; then
  echo "Missing ${SIGNAL}.freeze" >&2
  exit 20
fi
if [[ -e "${OUTPUT}" ]]; then
  echo "Refusing duplicate launch: ${OUTPUT} exists" >&2
  exit 21
fi
mapfile -t gpu_pids < <(
  nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)
if (( ${#gpu_pids[@]} )); then
  echo "GPU is busy: ${gpu_pids[*]}" >&2
  exit 22
fi
if ! (set -o noclobber; printf '{"experiment_id":"EXP-0005","started_at":"%s","pid":%d}\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$$" > "${SIGNAL}.started"); then
  echo "Refusing duplicate launch: ${SIGNAL}.started exists" >&2
  exit 23
fi

mkdir -p "${OUTPUT}"
cp "${SIGNAL}.freeze" "${OUTPUT}/.freeze"
cp "${SIGNAL}.started" "${OUTPUT}/.started"

fail() {
  local status=$1
  local phase=$2
  printf '{"experiment_id":"EXP-0005","failed_at":"%s","phase":"%s","exit_code":%d}\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${phase}" "${status}" \
    > "${SIGNAL}.failed"
  cp "${SIGNAL}.failed" "${OUTPUT}/.failed"
  exit "${status}"
}

cd "${RUNTIME}"
env PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
  "${PYTHON}" dreamerv3/main.py \
  --logdir "${OUTPUT}/train" \
  --configs dmc_proprio size12m \
  --task dmc_walker_walk \
  --run.script train \
  --seed 0 \
  --tensorboard False \
  --env.dmc.repeat 2 \
  --run.num_envs 16 \
  --run.steps 250000 \
  --run.train_ratio 512 \
  --run.save_every 600 \
  --run.save_at_end True \
  > "${OUTPUT}/train_stdout.log" 2>&1 || fail $? train

readonly CHECKPOINT=${OUTPUT}/train/checkpoint.ckpt
if [[ ! -s "${CHECKPOINT}" ]]; then
  echo "Missing terminal checkpoint: ${CHECKPOINT}" >&2
  fail 24 checkpoint
fi
checkpoint_step=$("${PYTHON}" -c \
  "import cloudpickle,pathlib; print(cloudpickle.loads(pathlib.Path('${CHECKPOINT}').read_bytes())['step'])") \
  || fail $? checkpoint_step
printf '%s\n' "${checkpoint_step}" > "${OUTPUT}/checkpoint_step.txt"
if [[ "${checkpoint_step}" != "250000" ]]; then
  echo "Expected checkpoint step 250000, got ${checkpoint_step}" >&2
  fail 25 checkpoint_step
fi
sha256sum "${CHECKPOINT}" > "${OUTPUT}/checkpoint.sha256"

printf '{"experiment_id":"EXP-0005","completed_at":"%s","exit_code":0,"checkpoint_step":250000}\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${SIGNAL}.completed"
cp "${SIGNAL}.completed" "${OUTPUT}/.completed"
