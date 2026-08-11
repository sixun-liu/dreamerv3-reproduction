#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <smoke|pilot|full>" >&2
  exit 2
fi

readonly STAGE=$1
readonly EXPERIMENT=EXP-0010
readonly CONTROL=/root/autodl-tmp/Code/DreamerV3/dreamerv3-reproduction
readonly RUNTIME=/root/autodl-tmp/Code/DreamerV3/dreamerv3-runtime-2411-crossdomain
readonly WORKFLOW=/root/autodl-tmp/Tools/research-agent-kit
readonly PYTHON=/root/autodl-tmp/Envs/dv3-2411/bin/python
readonly VERIFY=${CONTROL}/scripts/verify_exp0010_run.py
readonly EVALUATE=${CONTROL}/scripts/evaluate_exp0010_checkpoint.py
readonly FORMAL_TAG=EXP-0010__walker-vision__s000__staged-1m-env__20260811T151208Z
readonly SMOKE_TAG=EXP-0010__walker-vision__s31415__smoke-8192-dec__20260811T151208Z

case "${STAGE}" in
  smoke)
    readonly TAG=${SMOKE_TAG}
    readonly CONFIG=${CONTROL}/docs/reproduction/configs/exp0010_dmcvision_smoke_s31415_8192_dec.yaml
    readonly EXPECTED_CONFIG_SHA=29105b22334363c436e164f34c5c084e230720e64a7fa38b5cc438ae01184451
    readonly STEPS=8192
    readonly ENV_STEPS=16384
    readonly SEED=31415
    ;;
  pilot)
    readonly TAG=${FORMAL_TAG}
    readonly CONFIG=${CONTROL}/docs/reproduction/configs/exp0010_dmcvision_s000_100k_env.yaml
    readonly EXPECTED_CONFIG_SHA=ef1068f85184f0937aa512aecf60b01e3519a34726b193edec28c88e55330588
    readonly STEPS=50000
    readonly ENV_STEPS=100000
    readonly SEED=0
    ;;
  full)
    readonly TAG=${FORMAL_TAG}
    readonly CONFIG=${CONTROL}/docs/reproduction/configs/exp0010_dmcvision_s000_1m_env.yaml
    readonly EXPECTED_CONFIG_SHA=ab1a663629a43e0bf197296bb34ee9301bd85926f2970ebd7c9b4b86b5e0300f
    readonly STEPS=500000
    readonly ENV_STEPS=1000000
    readonly SEED=0
    ;;
  *)
    echo "Unknown stage: ${STAGE}" >&2
    exit 2
    ;;
esac

readonly ROOT=/root/autodl-tmp/Runs/${TAG}
readonly SIGNAL=/root/autodl-tmp/Runs/${TAG}
readonly STAGE_STARTED=${ROOT}/.${STAGE}.started
readonly STAGE_COMPLETED=${ROOT}/.${STAGE}.completed
readonly STAGE_FAILED=${ROOT}/.${STAGE}.failed
readonly STDOUT_LOG=${ROOT}/train_${STAGE}_stdout.log
readonly INTEGRITY=${ROOT}/integrity_${STAGE}.json
readonly START_EPOCH=$(date +%s)
SAMPLER_PID=

fail() {
  local status=$1
  local phase=$2
  printf \
    '{"experiment_id":"%s","stage":"%s","failed_at":"%s","phase":"%s","exit_code":%d,"wall_seconds":%d}\n' \
    "${EXPERIMENT}" "${STAGE}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    "${phase}" "${status}" "$(( $(date +%s) - START_EPOCH ))" > "${STAGE_FAILED}"
  cp "${STAGE_FAILED}" "${SIGNAL}.${STAGE}.failed"
  exit "${status}"
}

stop_sampler() {
  if [[ -n "${SAMPLER_PID}" ]]; then
    kill "${SAMPLER_PID}" 2>/dev/null || true
    wait "${SAMPLER_PID}" 2>/dev/null || true
    SAMPLER_PID=
  fi
}
trap stop_sampler EXIT

sample_resources() {
  local watched_pid=$1
  local gpu_csv=$2
  local process_csv=$3
  printf 'timestamp,memory_used_mib,gpu_util_percent,power_watts,temperature_c\n' > "${gpu_csv}"
  printf 'sampled_at,pid,process_name,used_memory_mib\n' > "${process_csv}"
  while kill -0 "${watched_pid}" 2>/dev/null; do
    nvidia-smi \
      --query-gpu=timestamp,memory.used,utilization.gpu,power.draw,temperature.gpu \
      --format=csv,noheader,nounits >> "${gpu_csv}" 2>/dev/null || true
    sampled_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    while IFS= read -r row; do
      [[ -z "${row}" ]] || printf '%s,%s\n' "${sampled_at}" "${row}" >> "${process_csv}"
    done < <(nvidia-smi \
      --query-compute-apps=pid,process_name,used_memory \
      --format=csv,noheader,nounits 2>/dev/null || true)
    sleep 30
  done
}

write_run_freeze() {
  local freeze_path=$1
  local control_commit runtime_commit workflow_commit pilot_sha full_sha
  control_commit=$(git -C "${CONTROL}" rev-parse HEAD)
  runtime_commit=$(git -C "${RUNTIME}" rev-parse HEAD)
  workflow_commit=$(git -C "${WORKFLOW}" rev-parse HEAD)
  pilot_sha=$(sha256sum "${CONTROL}/docs/reproduction/configs/exp0010_dmcvision_s000_100k_env.yaml" | cut -d' ' -f1)
  full_sha=$(sha256sum "${CONTROL}/docs/reproduction/configs/exp0010_dmcvision_s000_1m_env.yaml" | cut -d' ' -f1)
  "${PYTHON}" - "${freeze_path}" "${TAG}" "${STAGE}" \
    "${control_commit}" "${runtime_commit}" "${workflow_commit}" \
    "${CONFIG}" "${EXPECTED_CONFIG_SHA}" "${pilot_sha}" "${full_sha}" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

(
    path, tag, stage, control_commit, runtime_commit, workflow_commit,
    config, config_sha, pilot_sha, full_sha,
) = sys.argv[1:]
payload = {
    "schema_version": 1,
    "experiment_id": "EXP-0010",
    "tag": tag,
    "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "initial_stage": stage,
    "control_commit": control_commit,
    "runtime_commit": runtime_commit,
    "workflow_commit": workflow_commit,
    "config": config,
    "config_sha256": config_sha,
    "staged_config_sha256": {"pilot": pilot_sha, "full": full_sha},
    "seed_policy": "smoke=31415; formal=0; DMC environment RNG uncontrolled",
    "continuation_policy": "pilot gate only; same logdir/replay/checkpoint",
}
Path(path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY
}

if [[ ! -f "${CONFIG}" ]]; then
  echo "Missing frozen config: ${CONFIG}" >&2
  exit 20
fi
actual_config_sha=$(sha256sum "${CONFIG}" | cut -d' ' -f1)
if [[ "${actual_config_sha}" != "${EXPECTED_CONFIG_SHA}" ]]; then
  echo "Config hash drift: ${actual_config_sha}" >&2
  exit 21
fi
if [[ "$(git -C "${CONTROL}" status --porcelain)" ]]; then
  echo "Control repo must be clean before launch" >&2
  exit 22
fi
if [[ "$(git -C "${RUNTIME}" status --porcelain)" ]]; then
  echo "Runtime repo must be clean before launch" >&2
  exit 23
fi
if [[ "$(git -C "${RUNTIME}" rev-parse HEAD)" != "6642b941f578cd72147bc2be3c3343d5bc72931c" ]]; then
  echo "Runtime commit drift" >&2
  exit 24
fi
mapfile -t gpu_pids < <(
  nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits |
    sed '/^[[:space:]]*$/d')
if (( ${#gpu_pids[@]} )); then
  echo "GPU is busy: ${gpu_pids[*]}" >&2
  exit 25
fi
available_kib=$(df --output=avail /root/autodl-tmp | tail -1)
if (( available_kib < 5 * 1024 * 1024 )); then
  echo "Data disk has less than the frozen 5 GiB floor" >&2
  exit 26
fi

if [[ "${STAGE}" == "full" ]]; then
  if [[ ! -f "${ROOT}/.pilot.completed" || ! -f "${ROOT}/gate_100k.json" ]]; then
    echo "Pilot completion or 100K gate is missing" >&2
    exit 27
  fi
  if ! "${PYTHON}" - "${ROOT}/gate_100k.json" <<'PY'
import json
import sys
raise SystemExit(0 if json.load(open(sys.argv[1]))["continuation_gate"] else 1)
PY
  then
    echo "100K continuation gate did not pass" >&2
    exit 28
  fi
  if [[ ! -s "${ROOT}/train/checkpoint.ckpt" || ! -s "${ROOT}/checkpoint_100k.ckpt" ]]; then
    echo "Pilot checkpoint or immutable 100K copy is missing" >&2
    exit 29
  fi
  if [[ -e "${STAGE_STARTED}" || -e "${STAGE_COMPLETED}" ]]; then
    echo "Refusing duplicate full-stage launch" >&2
    exit 30
  fi
else
  if [[ -e "${ROOT}" || -e "${SIGNAL}.started" || -e "${SIGNAL}.freeze" ]]; then
    echo "Refusing duplicate ${STAGE} launch: ${ROOT}" >&2
    exit 31
  fi
  write_run_freeze "${SIGNAL}.freeze"
  mkdir -p "${ROOT}"
  cp "${SIGNAL}.freeze" "${ROOT}/.freeze"
  printf \
    '{"experiment_id":"%s","tag":"%s","stage":"%s","started_at":"%s","pid":%d}\n' \
    "${EXPERIMENT}" "${TAG}" "${STAGE}" \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$$" > "${SIGNAL}.started"
  cp "${SIGNAL}.started" "${ROOT}/.started"
fi

if ! (set -o noclobber; printf \
  '{"experiment_id":"%s","stage":"%s","started_at":"%s","pid":%d}\n' \
  "${EXPERIMENT}" "${STAGE}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$$" \
  > "${STAGE_STARTED}"); then
  echo "Refusing duplicate stage launch: ${STAGE_STARTED}" >&2
  exit 32
fi
cp "${STAGE_STARTED}" "${SIGNAL}.${STAGE}.started"

find /dev/shm -maxdepth 1 -type f \
  \( -name 'torch_*' -o -name 'sem.loky-*' -o -name 'cuda.shm.*' \) \
  -delete

{
  date -u +started_at=%Y-%m-%dT%H:%M:%SZ
  df -B1 /root/autodl-tmp
  nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free,utilization.gpu \
    --format=csv,noheader,nounits
} > "${ROOT}/resource_${STAGE}_before.txt"

cd "${RUNTIME}"
/usr/bin/time -v -o "${ROOT}/resource_${STAGE}_time.txt" \
  env PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
  "${PYTHON}" dreamerv3/main.py \
  --logdir "${ROOT}/train" \
  --configs dmc_vision size12m \
  --task dmc_walker_walk \
  --run.script train \
  --seed "${SEED}" \
  --tensorboard False \
  --env.dmc.repeat 2 \
  --run.num_envs 16 \
  --run.steps "${STEPS}" \
  --run.train_ratio 512 \
  --run.log_every 120 \
  --run.save_every 600 \
  --run.save_at_end True \
  > "${STDOUT_LOG}" 2>&1 &
TRAIN_PID=$!
sample_resources "${TRAIN_PID}" "${ROOT}/resource_${STAGE}_gpu.csv" \
  "${ROOT}/resource_${STAGE}_process.csv" &
SAMPLER_PID=$!
set +e
wait "${TRAIN_PID}"
train_status=$?
set -e
stop_sampler
if (( train_status != 0 )); then
  fail "${train_status}" train
fi

"${PYTHON}" "${VERIFY}" \
  --run-dir "${ROOT}" \
  --frozen-config "${CONFIG}" \
  --expected-step "${STEPS}" \
  --stdout-log "${STDOUT_LOG}" \
  --output "${INTEGRITY}" \
  > "${ROOT}/integrity_${STAGE}_stdout.log" 2>&1 || fail $? integrity

if [[ "${STAGE}" == "pilot" ]]; then
  cp --reflink=auto "${ROOT}/train/checkpoint.ckpt" "${ROOT}/checkpoint_100k.ckpt"
  sha256sum "${ROOT}/checkpoint_100k.ckpt" > "${ROOT}/checkpoint_100k.sha256"
elif [[ "${STAGE}" == "smoke" ]]; then
  cd "${CONTROL}"
  env PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
    "${PYTHON}" "${EVALUATE}" \
    --runtime "${RUNTIME}" \
    --checkpoint "${ROOT}/train/checkpoint.ckpt" \
    --output "${ROOT}/smoke_eval" \
    --episodes 1 \
    --agent-seed 27182 \
    > "${ROOT}/smoke_eval_stdout.log" 2>&1 || fail $? smoke_eval
fi

{
  date -u +completed_at=%Y-%m-%dT%H:%M:%SZ
  df -B1 /root/autodl-tmp
  du -sb "${ROOT}"
  nvidia-smi --query-gpu=name,memory.total,memory.free,utilization.gpu \
    --format=csv,noheader,nounits
} > "${ROOT}/resource_${STAGE}_after.txt"

readonly WALL_SECONDS=$(( $(date +%s) - START_EPOCH ))
readonly OUTPUT_BYTES=$(du -sb "${ROOT}" | awk '{print $1}')
printf \
  '{"experiment_id":"%s","stage":"%s","completed_at":"%s","exit_code":0,"checkpoint_step":%d,"environment_steps":%d,"wall_seconds":%d,"output_bytes":%d}\n' \
  "${EXPERIMENT}" "${STAGE}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  "${STEPS}" "${ENV_STEPS}" "${WALL_SECONDS}" "${OUTPUT_BYTES}" \
  > "${STAGE_COMPLETED}"
cp "${STAGE_COMPLETED}" "${SIGNAL}.${STAGE}.completed"

if [[ "${STAGE}" == "full" ]]; then
  cp "${STAGE_COMPLETED}" "${ROOT}/.completed"
  cp "${STAGE_COMPLETED}" "${SIGNAL}.completed"
elif [[ "${STAGE}" == "smoke" ]]; then
  cp "${STAGE_COMPLETED}" "${ROOT}/.completed"
  cp "${STAGE_COMPLETED}" "${SIGNAL}.completed"
fi
