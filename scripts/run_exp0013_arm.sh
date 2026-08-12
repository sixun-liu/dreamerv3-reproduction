#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || ! "$1" =~ ^(2|4|8)$ ]]; then
  echo "Usage: $0 <2|4|8>" >&2
  exit 2
fi

readonly ENVS=$1
readonly EXPERIMENT=EXP-0013
readonly CONTROL=/root/autodl-tmp/Code/DreamerV3/dreamerv3-reproduction
readonly RUNTIME=/root/autodl-tmp/Code/DreamerV3/dreamerv3-runtime-2026-crossdomain
readonly WORKFLOW=/root/autodl-tmp/Tools/research-agent-kit
readonly PYTHON=/root/autodl-tmp/Envs/dv3-minecraft-2026/bin/python
readonly RUNTIME_COMMIT=5168475b7a4413f9575933b4580e7073caea2114
readonly RUN_ROOT=/root/autodl-tmp/Runs/EXP-0013__minecraft-throughput__s31415__5040-env__20260812T060000Z
readonly ROOT=${RUN_ROOT}/envs${ENVS}
readonly CONFIG=${CONTROL}/docs/reproduction/configs/exp0013_minecraft_envs${ENVS}_s31415_5040_env.yaml
readonly MATRIX=${CONTROL}/docs/reproduction/configs/exp0013_minecraft_throughput_matrix.yaml
readonly STEPS=5040
readonly STARTED=${ROOT}.started
readonly COMPLETED=${ROOT}/.completed
readonly FAILED=${ROOT}/.failed
readonly STDOUT_LOG=${ROOT}/train_stdout.log
readonly START_EPOCH=$(date +%s)
SAMPLER_PID=

case "${ENVS}" in
  2) readonly DRIVER_STEP_QUANTUM=10 ;;
  4) readonly DRIVER_STEP_QUANTUM=12 ;;
  8) readonly DRIVER_STEP_QUANTUM=16 ;;
esac

stop_sampler() {
  if [[ -n "${SAMPLER_PID}" ]]; then
    kill "${SAMPLER_PID}" 2>/dev/null || true
    wait "${SAMPLER_PID}" 2>/dev/null || true
    SAMPLER_PID=
  fi
}

fail() {
  local status=$1
  local phase=$2
  stop_sampler
  printf \
    '{"experiment_id":"%s","arm":"envs%s","failed_at":"%s","phase":"%s","exit_code":%d,"wall_seconds":%d}\n' \
    "${EXPERIMENT}" "${ENVS}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    "${phase}" "${status}" "$(( $(date +%s) - START_EPOCH ))" > "${FAILED}"
  exit "${status}"
}
trap stop_sampler EXIT

if [[ ! -f "${RUN_ROOT}/l0-envs2-r1/.completed" ]]; then
  echo "EXP-0013 multi-environment L0 is missing" >&2
  exit 20
fi
if [[ -e "${ROOT}" || -e "${STARTED}" ]]; then
  echo "Refusing duplicate EXP-0013 envs${ENVS} launch" >&2
  exit 21
fi
if [[ "$(git -C "${CONTROL}" status --porcelain)" ]]; then
  echo "Control repo must be clean" >&2
  exit 22
fi
if [[ "$(git -C "${RUNTIME}" status --porcelain)" || \
      "$(git -C "${RUNTIME}" rev-parse HEAD)" != "${RUNTIME_COMMIT}" ]]; then
  echo "Runtime provenance drift" >&2
  exit 23
fi
"${PYTHON}" "${CONTROL}/scripts/generate_exp0013_configs.py" \
  --baseline "${CONTROL}/docs/reproduction/configs/exp0012_minecraft_smoke_s31415_4096_env.yaml" \
  --run-root "${RUN_ROOT}" \
  --output-dir "${CONTROL}/docs/reproduction/configs" --check >/dev/null
if [[ "${ENVS}" -ge 4 ]]; then
  readonly PREV=${RUN_ROOT}/envs2/analysis.json
  if [[ ! -f "${PREV}" ]] || ! "${PYTHON}" -c \
    'import json,sys; raise SystemExit(0 if json.load(open(sys.argv[1]))["expand_recommended"] else 1)' \
    "${PREV}"; then
    echo "envs2 did not authorize expansion" >&2
    exit 24
  fi
fi
if [[ "${ENVS}" -eq 8 ]]; then
  readonly CURRENT=${RUN_ROOT}/envs4/analysis.json
  if [[ ! -f "${CURRENT}" ]] || ! "${PYTHON}" -c \
    'import json,sys; a=json.load(open(sys.argv[1])); b=json.load(open(sys.argv[2])); ok=b["expand_recommended"] and b["policy_fps_tail_mean"] >= 1.05*a["policy_fps_tail_mean"] and b["resource"]["max_cgroup_memory_bytes"] <= 75161927680; raise SystemExit(0 if ok else 1)' \
    "${PREV}" "${CURRENT}"; then
    echo "envs4 did not authorize envs8" >&2
    exit 25
  fi
fi
mapfile -t gpu_pids < <(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d')
if (( ${#gpu_pids[@]} )); then
  echo "GPU is busy: ${gpu_pids[*]}" >&2
  exit 26
fi
if (( $(df --output=avail -B1 / | tail -1) < 5 * 1024 * 1024 * 1024 )); then
  echo "System disk has less than 5 GiB free" >&2
  exit 27
fi
if (( $(df --output=avail -B1 /root/autodl-tmp | tail -1) < 10 * 1024 * 1024 * 1024 )); then
  echo "Data disk has less than 10 GiB free" >&2
  exit 28
fi

mkdir -p "${ROOT}/work/tmp" "${ROOT}/work/malmo" "${ROOT}/work/jax-cache" \
  "${ROOT}/work/cache" "${ROOT}/work/mesa-cache" "${ROOT}/work/cuda-cache"
printf '{"experiment_id":"%s","arm":"envs%s","started_at":"%s","pid":%d}\n' \
  "${EXPERIMENT}" "${ENVS}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$$" > "${STARTED}"
cp "${STARTED}" "${ROOT}/.started"
printf \
  '{"schema_version":1,"experiment_id":"%s","arm":"envs%s","control_commit":"%s","runtime_commit":"%s","workflow_commit":"%s","config":"%s","config_sha256":"%s","matrix_sha256":"%s","seed":31415,"step_unit":"environment steps"}\n' \
  "${EXPERIMENT}" "${ENVS}" "$(git -C "${CONTROL}" rev-parse HEAD)" \
  "${RUNTIME_COMMIT}" "$(git -C "${WORKFLOW}" rev-parse HEAD)" "${CONFIG}" \
  "$(sha256sum "${CONFIG}" | cut -d' ' -f1)" "$(sha256sum "${MATRIX}" | cut -d' ' -f1)" \
  > "${ROOT}/.freeze"
"${PYTHON}" -m pip freeze > "${ROOT}/python_environment.txt"

find /dev/shm -maxdepth 1 -type f \
  \( -name 'torch_*' -o -name 'sem.loky-*' -o -name 'cuda.shm.*' \) -delete
cd "${ROOT}/work"
readonly TRAIN_START_EPOCH=$(date +%s)
/usr/bin/time -v -o "${ROOT}/resource_time.txt" \
  env \
    TMPDIR="${ROOT}/work/tmp" TMP="${ROOT}/work/tmp" TEMP="${ROOT}/work/tmp" \
    MALMO_MINECRAFT_OUTPUT_LOGDIR="${ROOT}/work/malmo" \
    JAX_COMPILATION_CACHE_DIR="${ROOT}/work/jax-cache" \
    XDG_CACHE_HOME="${ROOT}/work/cache" MESA_SHADER_CACHE_DIR="${ROOT}/work/mesa-cache" \
    CUDA_CACHE_PATH="${ROOT}/work/cuda-cache" \
    PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    timeout --signal=TERM --kill-after=60s 1800s \
    xvfb-run -a -s '-screen 0 1024x768x24 -ac +extension GLX +render -noreset' \
    "${PYTHON}" "${RUNTIME}/dreamerv3/main.py" \
    --logdir "${ROOT}/train" --configs minecraft size50m \
    --task minecraft_diamond --script train --seed 31415 \
    --run.envs "${ENVS}" --run.debug False --run.steps "${STEPS}" \
    --run.train_ratio 32 --run.log_every 10 --run.report_every 120 \
    --run.save_every 300 --run.save_at_end True \
    > "${STDOUT_LOG}" 2>&1 &
TRAIN_PID=$!
"${PYTHON}" "${CONTROL}/scripts/sample_exp0013_resources.py" \
  --watched-pid "${TRAIN_PID}" --output-dir "${ROOT}" \
  --temp-root "${ROOT}/work/tmp" --interval 5 &
SAMPLER_PID=$!
set +e
wait "${TRAIN_PID}"
train_status=$?
set -e
stop_sampler
if (( train_status != 0 )); then
  fail "${train_status}" train
fi
readonly TRAIN_WALL_SECONDS=$(( $(date +%s) - TRAIN_START_EPOCH ))
"${PYTHON}" "${CONTROL}/scripts/cleanup_exp0013_temp.py" \
  --temp-root "${ROOT}/work/tmp" --output "${ROOT}/temp_cleanup_postprocess.json" \
  --wait-seconds 30 > "${ROOT}/temp_cleanup_postprocess_stdout.log" 2>&1 \
  || fail $? temp_cleanup

"${PYTHON}" "${CONTROL}/scripts/verify_exp0013_throughput.py" \
  --run-dir "${ROOT}" --frozen-config "${CONFIG}" \
  --expected-step "${STEPS}" --expected-envs "${ENVS}" \
  --driver-step-quantum "${DRIVER_STEP_QUANTUM}" --stdout-log "${STDOUT_LOG}" \
  --resource-system "${ROOT}/resource_system.csv" --temp-root "${ROOT}/work/tmp" \
  --output "${ROOT}/integrity.json" > "${ROOT}/integrity_stdout.log" 2>&1 \
  || fail $? integrity

readonly WALL_SECONDS=$(( $(date +%s) - START_EPOCH ))
"${PYTHON}" "${CONTROL}/scripts/analyze_exp0013_throughput.py" \
  --run-dir "${ROOT}" --envs "${ENVS}" --steps "${STEPS}" \
  --wall-seconds "${TRAIN_WALL_SECONDS}" --historical-wall-fps 24.67308166790032 \
  --historical-policy-fps 25.50 --expand-min-policy-speedup 1.10 \
  --target-policy-speedup 1.50 --output "${ROOT}/analysis.json" \
  > "${ROOT}/analysis_stdout.log" 2>&1 || fail $? analysis
printf \
  '{"experiment_id":"%s","arm":"envs%s","completed_at":"%s","exit_code":0,"requested_environment_steps":%d,"training_wall_seconds":%d,"total_wall_seconds":%d,"output_bytes":%d}\n' \
  "${EXPERIMENT}" "${ENVS}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  "${STEPS}" "${TRAIN_WALL_SECONDS}" "${WALL_SECONDS}" \
  "$(du -sb "${ROOT}" | awk '{print $1}')" \
  > "${COMPLETED}"
