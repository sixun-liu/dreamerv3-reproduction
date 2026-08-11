#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <smoke|formal>" >&2
  exit 2
fi

readonly STAGE=$1
readonly EXPERIMENT=EXP-0011
readonly CONTROL=/root/autodl-tmp/Code/DreamerV3/dreamerv3-reproduction
readonly RUNTIME=/root/autodl-tmp/Code/DreamerV3/dreamerv3-runtime-2026-crossdomain
readonly WORKFLOW=/root/autodl-tmp/Tools/research-agent-kit
readonly PYTHON=/root/autodl-tmp/Envs/dv3-atari-2026/bin/python
readonly ROMDIR=/root/autodl-tmp/ThirdParty/atari-roms
readonly REFERENCE=${RUNTIME}/scores/atari100k-dreamerv3.json.gz
readonly VERIFY=${CONTROL}/scripts/verify_exp0011_run.py
readonly ANALYZE=${CONTROL}/scripts/analyze_exp0011_atari.py
readonly L0_TAG=EXP-0011__breakout__s31415__ale-l0__20260811T201000Z
readonly SMOKE_TAG=EXP-0011__breakout__s31415__smoke-r1-2048-dec__20260811T202700Z
readonly FORMAL_TAG=EXP-0011__breakout__s000__100k-dec__20260811T201000Z

case "${STAGE}" in
  smoke)
    readonly TAG=${SMOKE_TAG}
    readonly CONFIG=${CONTROL}/docs/reproduction/configs/exp0011_atari100k_smoke_r1_s31415_2048_dec.yaml
    readonly EXPECTED_CONFIG_SHA=5f3fcc1ea35feaeb83898904de1f30e4e48f91fded13d43ef2a50348ad594d5a
    readonly STEPS=2048
    readonly SEED=31415
    ;;
  formal)
    readonly TAG=${FORMAL_TAG}
    readonly CONFIG=${CONTROL}/docs/reproduction/configs/exp0011_atari100k_s000_100k_dec.yaml
    readonly EXPECTED_CONFIG_SHA=014da22ccad6427836fa6b1b7af40d0e49758cd81d1f1b25fc41aef5024ed7f4
    readonly STEPS=100000
    readonly SEED=0
    ;;
  *)
    echo "Unknown stage: ${STAGE}" >&2
    exit 2
    ;;
esac

readonly ROOT=/root/autodl-tmp/Runs/${TAG}
readonly STARTED=${ROOT}.started
readonly COMPLETED=${ROOT}.completed
readonly FAILED=${ROOT}.failed
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
  rm -f "${STAGE_COMPLETED}" "${COMPLETED}"
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
    sleep 30
  done
}

if [[ ! -f "${CONFIG}" || ! -x "${PYTHON}" || ! -s "${ROMDIR}/breakout.bin" ]]; then
  echo "Frozen config, Atari environment, or ROM is missing" >&2
  exit 20
fi
if [[ "$(sha256sum "${CONFIG}" | cut -d' ' -f1)" != "${EXPECTED_CONFIG_SHA}" ]]; then
  echo "Config hash drift" >&2
  exit 21
fi
if [[ "$(md5sum "${ROMDIR}/breakout.bin" | cut -d' ' -f1)" != "f34f08e5eb96e500e851a80be3277a56" ]]; then
  echo "Breakout ROM fingerprint drift" >&2
  exit 22
fi
if [[ "$(git -C "${CONTROL}" status --porcelain)" ]]; then
  echo "Control repo must be clean before launch" >&2
  exit 23
fi
if [[ "$(git -C "${RUNTIME}" status --porcelain)" || \
      "$(git -C "${RUNTIME}" rev-parse HEAD)" != "5168475b7a4413f9575933b4580e7073caea2114" ]]; then
  echo "Runtime provenance drift" >&2
  exit 24
fi
mapfile -t gpu_pids < <(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d')
if (( ${#gpu_pids[@]} )); then
  echo "GPU is busy: ${gpu_pids[*]}" >&2
  exit 25
fi
if (( $(df --output=avail /root/autodl-tmp | tail -1) < 5 * 1024 * 1024 )); then
  echo "Data disk has less than 5 GiB free" >&2
  exit 26
fi
if [[ ! -f "/root/autodl-tmp/Runs/${L0_TAG}.completed" ]]; then
  echo "ALE L0 completion is missing" >&2
  exit 27
fi
if [[ "${STAGE}" == "formal" ]]; then
  readonly SMOKE_ROOT=/root/autodl-tmp/Runs/${SMOKE_TAG}
  if [[ ! -f "${SMOKE_ROOT}/.smoke.completed" || ! -f "${SMOKE_ROOT}/gate_smoke.json" ]]; then
    echo "Smoke completion or gate is missing" >&2
    exit 28
  fi
  if ! "${PYTHON}" -c 'import json,sys; raise SystemExit(0 if json.load(open(sys.argv[1]))["formal_gate"] else 1)' \
    "${SMOKE_ROOT}/gate_smoke.json"; then
    echo "Smoke formal gate did not pass" >&2
    exit 29
  fi
fi
if [[ -e "${ROOT}" || -e "${STARTED}" ]]; then
  echo "Refusing duplicate EXP-0011 ${STAGE} launch" >&2
  exit 30
fi

mkdir -p "${ROOT}"
printf '{"experiment_id":"%s","tag":"%s","stage":"%s","started_at":"%s","pid":%d}\n' \
  "${EXPERIMENT}" "${TAG}" "${STAGE}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$$" > "${STARTED}"
cp "${STARTED}" "${ROOT}/.started"
cp "${STARTED}" "${STAGE_STARTED}"
printf \
  '{"schema_version":1,"experiment_id":"%s","tag":"%s","stage":"%s","control_commit":"%s","runtime_commit":"%s","workflow_commit":"%s","config":"%s","config_sha256":"%s","rom_md5":"f34f08e5eb96e500e851a80be3277a56","seed":%d,"step_unit":"agent decisions; logger x-axis uses repeat4 emulator frames"}\n' \
  "${EXPERIMENT}" "${TAG}" "${STAGE}" "$(git -C "${CONTROL}" rev-parse HEAD)" \
  "$(git -C "${RUNTIME}" rev-parse HEAD)" "$(git -C "${WORKFLOW}" rev-parse HEAD)" \
  "${CONFIG}" "${EXPECTED_CONFIG_SHA}" "${SEED}" > "${ROOT}.freeze"
cp "${ROOT}.freeze" "${ROOT}/.freeze"

find /dev/shm -maxdepth 1 -type f \
  \( -name 'torch_*' -o -name 'sem.loky-*' -o -name 'cuda.shm.*' \) -delete
{
  date -u +started_at=%Y-%m-%dT%H:%M:%SZ
  df -B1 /root/autodl-tmp
  nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free,utilization.gpu \
    --format=csv,noheader,nounits
} > "${ROOT}/resource_${STAGE}_before.txt"

cd "${RUNTIME}"
/usr/bin/time -v -o "${ROOT}/resource_${STAGE}_time.txt" \
  env ALE_ROM_PATH="${ROMDIR}" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  "${PYTHON}" dreamerv3/main.py \
  --logdir "${ROOT}/train" \
  --configs atari100k size50m \
  --task atari100k_breakout \
  --script train \
  --seed "${SEED}" \
  --run.envs 1 \
  --run.steps "${STEPS}" \
  --run.train_ratio 256 \
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
printf \
  '{"experiment_id":"%s","stage":"%s","completed_at":"%s","exit_code":0,"checkpoint_step":%d,"agent_decisions":%d,"emulator_frames":%d,"wall_seconds":%d,"output_bytes":%d}\n' \
  "${EXPERIMENT}" "${STAGE}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  "${STEPS}" "${STEPS}" "$(( STEPS * 4 ))" "${WALL_SECONDS}" "${OUTPUT_BYTES}" > "${STAGE_COMPLETED}"

if [[ "${STAGE}" == "smoke" ]]; then
  "${PYTHON}" "${ANALYZE}" \
    --stage smoke \
    --run-dir "${ROOT}" \
    --reference "${REFERENCE}" \
    --output-dir /root/autodl-tmp/Artifacts/dreamerv3/EXP-0011 \
    > "${ROOT}/analysis_smoke_stdout.log" 2>&1 || fail $? analysis
fi
cp "${STAGE_COMPLETED}" "${COMPLETED}"
cp "${STAGE_COMPLETED}" "${ROOT}/.completed"
