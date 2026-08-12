#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <smoke|formal>" >&2
  exit 2
fi

readonly STAGE=$1
readonly EXPERIMENT=EXP-0012
readonly CONTROL=/root/autodl-tmp/Code/DreamerV3/dreamerv3-reproduction
readonly RUNTIME=/root/autodl-tmp/Code/DreamerV3/dreamerv3-runtime-2026-crossdomain
readonly WORKFLOW=/root/autodl-tmp/Tools/research-agent-kit
readonly PYTHON=/root/autodl-tmp/Envs/dv3-minecraft-2026/bin/python
readonly VERIFY=${CONTROL}/scripts/verify_exp0012_run.py
readonly SMOKE_ANALYZE=${CONTROL}/scripts/analyze_exp0012_smoke.py
readonly L0_TAG=EXP-0012__minecraft-diamond__s31415__l0-32-step__20260812T080000Z
readonly SMOKE_TAG=EXP-0012__minecraft-diamond__s31415__smoke-4096-env__20260812T080000Z
readonly FORMAL_TAG=EXP-0012__minecraft-diamond__s000__100k-env__20260812T080000Z
readonly RUNTIME_COMMIT=5168475b7a4413f9575933b4580e7073caea2114

case "${STAGE}" in
  smoke)
    readonly TAG=${SMOKE_TAG}
    readonly CONFIG=${CONTROL}/docs/reproduction/configs/exp0012_minecraft_smoke_s31415_4096_env.yaml
    readonly STEPS=4096
    readonly SEED=31415
    readonly LOG_EVERY=30
    readonly REPORT_EVERY=120
    readonly SAVE_EVERY=300
    ;;
  formal)
    readonly TAG=${FORMAL_TAG}
    readonly CONFIG=${CONTROL}/docs/reproduction/configs/exp0012_minecraft_s000_100k_env.yaml
    readonly STEPS=100000
    readonly SEED=0
    readonly LOG_EVERY=120
    readonly REPORT_EVERY=300
    readonly SAVE_EVERY=900
    ;;
  *)
    echo "Unknown stage: ${STAGE}" >&2
    exit 2
    ;;
esac

readonly EXPECTED_CONFIG_SHA=$(sha256sum "${CONFIG}" | cut -d' ' -f1)
readonly ROOT=/root/autodl-tmp/Runs/${TAG}
readonly STARTED=${ROOT}.started
readonly COMPLETED=${ROOT}.completed
readonly FAILED=${ROOT}.failed
readonly STAGE_STARTED=${ROOT}/.${STAGE}.started
readonly STAGE_COMPLETED=${ROOT}/.${STAGE}.completed
readonly STAGE_FAILED=${ROOT}/.${STAGE}.failed
readonly STDOUT_LOG=${ROOT}/train_${STAGE}_stdout.log
readonly INTEGRITY=${ROOT}/integrity_${STAGE}.json
readonly DRIVER_STEP_QUANTUM=10
readonly START_EPOCH=$(date +%s)
SAMPLER_PID=

fail() {
  local status=$1
  local phase=$2
  printf \
    '{"experiment_id":"%s","stage":"%s","failed_at":"%s","phase":"%s","exit_code":%d,"wall_seconds":%d}\n' \
    "${EXPERIMENT}" "${STAGE}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    "${phase}" "${status}" "$(( $(date +%s) - START_EPOCH ))" > "${STAGE_FAILED}"
  cp "${STAGE_FAILED}" "${FAILED}"
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
    done < <(nvidia-smi --query-compute-apps=pid,process_name,used_memory \
      --format=csv,noheader,nounits 2>/dev/null || true)
    sleep 15
  done
}

if [[ ! -f "${CONFIG}" || ! -x "${PYTHON}" ]]; then
  echo "Frozen config or Minecraft environment is missing" >&2
  exit 20
fi
if [[ "$(git -C "${CONTROL}" status --porcelain)" ]]; then
  echo "Control repo must be clean before launch" >&2
  exit 21
fi
if [[ "$(git -C "${RUNTIME}" status --porcelain)" || \
      "$(git -C "${RUNTIME}" rev-parse HEAD)" != "${RUNTIME_COMMIT}" ]]; then
  echo "Runtime provenance drift" >&2
  exit 22
fi
if ! command -v java >/dev/null || ! command -v xvfb-run >/dev/null; then
  echo "Java or Xvfb is missing" >&2
  exit 23
fi
mapfile -t gpu_pids < <(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d')
if (( ${#gpu_pids[@]} )); then
  echo "GPU is busy: ${gpu_pids[*]}" >&2
  exit 24
fi
if (( $(df --output=avail /root/autodl-tmp | tail -1) < 5 * 1024 * 1024 )); then
  echo "Data disk has less than 5 GiB free" >&2
  exit 25
fi
if [[ ! -f "/root/autodl-tmp/Runs/${L0_TAG}.completed" ]]; then
  echo "Minecraft L0 completion is missing" >&2
  exit 26
fi
if [[ "${STAGE}" == "formal" ]]; then
  readonly SMOKE_ROOT=/root/autodl-tmp/Runs/${SMOKE_TAG}
  if [[ ! -f "${SMOKE_ROOT}/.smoke.completed" || ! -f "${SMOKE_ROOT}/gate_smoke.json" ]]; then
    echo "Smoke completion or gate is missing" >&2
    exit 27
  fi
  if ! "${PYTHON}" -c 'import json,sys; raise SystemExit(0 if json.load(open(sys.argv[1]))["formal_gate"] else 1)' \
    "${SMOKE_ROOT}/gate_smoke.json"; then
    echo "Smoke formal gate did not pass" >&2
    exit 28
  fi
  if [[ -f "${SMOKE_ROOT}/.smoke.failed" ]]; then
    readonly RECONCILIATION=${SMOKE_ROOT}/smoke_reconciliation.json
    if [[ ! -f "${RECONCILIATION}" ]] || ! "${PYTHON}" -c \
      'import json,sys; x=json.load(open(sys.argv[1])); raise SystemExit(0 if x["formal_gate"] and x["original_failure_retained"] and not x["scientific_inputs_changed"] else 1)' \
      "${RECONCILIATION}"; then
      echo "Retained smoke failure requires a valid reconciliation" >&2
      exit 30
    fi
  fi
fi
if [[ -e "${ROOT}" || -e "${STARTED}" ]]; then
  echo "Refusing duplicate EXP-0012 ${STAGE} launch" >&2
  exit 29
fi

mkdir -p "${ROOT}/work"
printf '{"experiment_id":"%s","tag":"%s","stage":"%s","started_at":"%s","pid":%d}\n' \
  "${EXPERIMENT}" "${TAG}" "${STAGE}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$$" > "${STARTED}"
cp "${STARTED}" "${ROOT}/.started"
cp "${STARTED}" "${STAGE_STARTED}"
printf \
  '{"schema_version":1,"experiment_id":"%s","tag":"%s","stage":"%s","control_commit":"%s","runtime_commit":"%s","workflow_commit":"%s","config":"%s","config_sha256":"%s","seed":%d,"step_unit":"environment steps"}\n' \
  "${EXPERIMENT}" "${TAG}" "${STAGE}" "$(git -C "${CONTROL}" rev-parse HEAD)" \
  "${RUNTIME_COMMIT}" "$(git -C "${WORKFLOW}" rev-parse HEAD)" \
  "${CONFIG}" "${EXPECTED_CONFIG_SHA}" "${SEED}" > "${ROOT}.freeze"
cp "${ROOT}.freeze" "${ROOT}/.freeze"
"${PYTHON}" -m pip freeze > "${ROOT}/python_environment.txt"

find /dev/shm -maxdepth 1 -type f \
  \( -name 'torch_*' -o -name 'sem.loky-*' -o -name 'cuda.shm.*' \) -delete
{
  date -u +started_at=%Y-%m-%dT%H:%M:%SZ
  df -B1 /root/autodl-tmp
  free -b
  nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free,utilization.gpu \
    --format=csv,noheader,nounits
} > "${ROOT}/resource_${STAGE}_before.txt"

cd "${ROOT}/work"
/usr/bin/time -v -o "${ROOT}/resource_${STAGE}_time.txt" \
  env PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  timeout --signal=TERM --kill-after=60s 43200s \
  xvfb-run -a -s '-screen 0 1024x768x24 -ac +extension GLX +render -noreset' \
  "${PYTHON}" "${RUNTIME}/dreamerv3/main.py" \
  --logdir "${ROOT}/train" \
  --configs minecraft size50m \
  --task minecraft_diamond \
  --script train \
  --seed "${SEED}" \
  --run.envs 1 \
  --run.steps "${STEPS}" \
  --run.train_ratio 32 \
  --run.log_every "${LOG_EVERY}" \
  --run.report_every "${REPORT_EVERY}" \
  --run.save_every "${SAVE_EVERY}" \
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

verify_args=(
  --run-dir "${ROOT}"
  --frozen-config "${CONFIG}"
  --expected-step "${STEPS}"
  --driver-step-quantum "${DRIVER_STEP_QUANTUM}"
  --stdout-log "${STDOUT_LOG}"
  --output "${INTEGRITY}"
)
if [[ "${STAGE}" == "formal" ]]; then
  verify_args+=(--require-scores)
fi
"${PYTHON}" "${VERIFY}" "${verify_args[@]}" \
  > "${ROOT}/integrity_${STAGE}_stdout.log" 2>&1 || fail $? integrity

{
  date -u +completed_at=%Y-%m-%dT%H:%M:%SZ
  df -B1 /root/autodl-tmp
  du -sb "${ROOT}"
  nvidia-smi --query-gpu=name,memory.total,memory.free,utilization.gpu \
    --format=csv,noheader,nounits
} > "${ROOT}/resource_${STAGE}_after.txt"
readonly WALL_SECONDS=$(( $(date +%s) - START_EPOCH ))
readonly OUTPUT_BYTES=$(du -sb "${ROOT}" | awk '{print $1}')
readonly DISK_FREE_BYTES=$(df --output=avail -B1 /root/autodl-tmp | tail -1)
readonly CHECKPOINT_STEP=$("${PYTHON}" -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint_step"])' "${INTEGRITY}")

if [[ "${STAGE}" == "smoke" ]]; then
  "${PYTHON}" "${SMOKE_ANALYZE}" \
    --integrity "${INTEGRITY}" \
    --gpu-csv "${ROOT}/resource_${STAGE}_gpu.csv" \
    --wall-seconds "${WALL_SECONDS}" \
    --output-bytes "${OUTPUT_BYTES}" \
    --disk-free-bytes "${DISK_FREE_BYTES}" \
    --output "${ROOT}/gate_smoke.json" \
    > "${ROOT}/gate_smoke_stdout.log" 2>&1
fi

printf \
  '{"experiment_id":"%s","stage":"%s","completed_at":"%s","exit_code":0,"requested_environment_steps":%d,"checkpoint_step":%d,"environment_steps":%d,"driver_step_quantum":%d,"wall_seconds":%d,"output_bytes":%d}\n' \
  "${EXPERIMENT}" "${STAGE}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  "${STEPS}" "${CHECKPOINT_STEP}" "${CHECKPOINT_STEP}" "${DRIVER_STEP_QUANTUM}" \
  "${WALL_SECONDS}" "${OUTPUT_BYTES}" > "${STAGE_COMPLETED}"
cp "${STAGE_COMPLETED}" "${COMPLETED}"
cp "${STAGE_COMPLETED}" "${ROOT}/.completed"
