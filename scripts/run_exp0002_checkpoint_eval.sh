#!/usr/bin/env bash
set -uo pipefail

readonly OUTPUT=/root/autodl-tmp/runs/EXP-0002__walker_walk__eval-s10000__40k-dec__20260721T050407Z
readonly RUNTIME=/root/autodl-tmp/dreamerv3
readonly PYTHON=/root/miniconda3/envs/dv3/bin/python
readonly CHECKPOINT=/root/autodl-tmp/runs/dv3_dmcp_walker_500k_s0_0717/ckpt/20260717T021720F580890

mkdir -p "${OUTPUT}"
if [[ ! -f "${OUTPUT}/.freeze" ]]; then
  echo "Missing ${OUTPUT}/.freeze" >&2
  exit 20
fi
if ! (set -o noclobber; printf '{"experiment_id":"EXP-0002","started_at":"%s","pid":%d}\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$$" > "${OUTPUT}/.started"); then
  echo "Refusing duplicate launch: ${OUTPUT}/.started exists" >&2
  exit 21
fi

cd "${RUNTIME}"
env PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  "${PYTHON}" dreamerv3/main.py \
  --logdir "${OUTPUT}" \
  --configs dmc_proprio size12m \
  --task dmc_walker_walk \
  --script eval_only \
  --seed 10000 \
  --env.dmc.repeat 2 \
  --run.envs 16 \
  --run.steps 40000 \
  --run.train_ratio 512 \
  --run.save_every 600 \
  --run.from_checkpoint "${CHECKPOINT}" \
  > "${OUTPUT}/stdout.log" 2>&1
status=$?

if [[ ${status} -eq 0 ]]; then
  printf '{"experiment_id":"EXP-0002","completed_at":"%s","exit_code":0}\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${OUTPUT}/.completed"
else
  printf '{"experiment_id":"EXP-0002","failed_at":"%s","exit_code":%d}\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${status}" > "${OUTPUT}/.failed"
fi
exit "${status}"
