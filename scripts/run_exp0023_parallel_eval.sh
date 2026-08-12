#!/usr/bin/env bash
set -euo pipefail

readonly EXPERIMENT=EXP-0023
readonly CONTROL=/root/autodl-tmp/Code/DreamerV3/dreamerv3-reproduction
readonly RUNTIME=/root/autodl-tmp/Code/DreamerV3/dreamerv3-exp0018-exact-stop
readonly RUNTIME_COMMIT=6723fc1620636cc8a52e0ddc0b7a9cbe946625b0
readonly PYTHON=/root/autodl-tmp/Envs/dv3-minecraft-2026/bin/python
readonly TRAIN=/root/autodl-tmp/Runs/EXP-0023__minecraft-diamond__s000__200k-to-500k-env__20260812T123000Z
readonly CHECKPOINT_ROOT=${TRAIN}/train/ckpt
readonly ROOT=/root/autodl-tmp/Runs/EXP-0023__minecraft-parallel-eval__s10000__3eps__20260812T123000Z
readonly STARTED=${ROOT}.started
readonly FAILED=${ROOT}/.failed
readonly START_EPOCH=$(date +%s)
SAMPLER_PID=

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
  trap - ERR
  mkdir -p "${ROOT}"
  rm -f "${ROOT}/.completed"
  printf \
    '{"experiment_id":"%s","failed_at":"%s","phase":"%s","exit_code":%d,"wall_seconds":%d}\n' \
    "${EXPERIMENT}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${phase}" \
    "${status}" "$(( $(date +%s) - START_EPOCH ))" > "${FAILED}"
  exit "${status}"
}
trap stop_sampler EXIT

if [[ -e "${ROOT}" || -e "${STARTED}" ]]; then
  echo "Refusing duplicate ${EXPERIMENT} evaluation launch" >&2
  exit 20
fi
if [[ "$(git -C "${CONTROL}" status --porcelain)" ]]; then
  echo "Control repo must be clean" >&2
  exit 21
fi
if [[ "$(git -C "${RUNTIME}" status --porcelain)" || \
      "$(git -C "${RUNTIME}" rev-parse HEAD)" != "${RUNTIME_COMMIT}" ]]; then
  echo "Runtime provenance drift" >&2
  exit 22
fi
if [[ ! -f "${TRAIN}/.completed" || ! -f "${TRAIN}/integrity.json" ]]; then
  echo "EXP-0023 training evidence is incomplete" >&2
  exit 23
fi
"${PYTHON}" -c 'import json,sys; assert json.load(open(sys.argv[1]))["passed"]' \
  "${TRAIN}/integrity.json"
checkpoint_name=$(<"${CHECKPOINT_ROOT}/latest")
if [[ ! "${checkpoint_name}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "Invalid checkpoint name" >&2
  exit 23
fi
readonly CHECKPOINT=${CHECKPOINT_ROOT}/${checkpoint_name}
for filename in agent.pkl step.pkl done; do
  if [[ ! -f "${CHECKPOINT}/${filename}" ]]; then
    echo "Incomplete checkpoint: ${filename}" >&2
    exit 23
  fi
done
readonly AGENT_SHA=$(sha256sum "${CHECKPOINT}/agent.pkl" | cut -d' ' -f1)
readonly STEP_SHA=$(sha256sum "${CHECKPOINT}/step.pkl" | cut -d' ' -f1)
mapfile -t gpu_pids < <(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d')
if (( ${#gpu_pids[@]} )); then
  echo "GPU is busy: ${gpu_pids[*]}" >&2
  exit 24
fi
if ps -eo comm= | awk '$1 == "java" || $1 == "Xvfb" {found=1} END {exit !found}'; then
  echo "Java or Xvfb process is already active" >&2
  exit 25
fi
if (( $(df --output=avail -B1 / | tail -1) < 5 * 1024 * 1024 * 1024 )); then
  echo "System disk has less than 5 GiB free" >&2
  exit 26
fi
if (( $(df --output=avail -B1 /root/autodl-tmp | tail -1) < 10 * 1024 * 1024 * 1024 )); then
  echo "Data disk has less than 10 GiB free" >&2
  exit 27
fi

mkdir -p "${ROOT}/work/tmp" "${ROOT}/work/malmo" "${ROOT}/work/jax-cache" \
  "${ROOT}/work/cache" "${ROOT}/work/mesa-cache" "${ROOT}/work/cuda-cache"
trap 'fail $? unexpected' ERR
printf '{"experiment_id":"%s","started_at":"%s","pid":%d}\n' \
  "${EXPERIMENT}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$$" > "${STARTED}"
cp "${STARTED}" "${ROOT}/.started"
printf \
  '{"schema_version":1,"experiment_id":"%s","control_commit":"%s","runtime_commit":"%s","checkpoint":"%s","checkpoint_agent_sha256":"%s","checkpoint_step_sha256":"%s","agent_seed":10000,"environment_count":3,"episodes_per_environment":1,"episode_length":36000,"video_worker":0,"video_stride":4}\n' \
  "${EXPERIMENT}" "$(git -C "${CONTROL}" rev-parse HEAD)" "${RUNTIME_COMMIT}" \
  "${CHECKPOINT}" "${AGENT_SHA}" "${STEP_SHA}" > "${ROOT}/.freeze"

find /dev/shm -maxdepth 1 -type f \
  \( -name 'torch_*' -o -name 'sem.loky-*' -o -name 'cuda.shm.*' \) -delete
cd "${ROOT}/work"
/usr/bin/time -v -o "${ROOT}/resource_time.txt" \
  env \
    TMPDIR="${ROOT}/work/tmp" TMP="${ROOT}/work/tmp" TEMP="${ROOT}/work/tmp" \
    MALMO_MINECRAFT_OUTPUT_LOGDIR="${ROOT}/work/malmo" \
    JAX_COMPILATION_CACHE_DIR="${ROOT}/work/jax-cache" \
    XDG_CACHE_HOME="${ROOT}/work/cache" MESA_SHADER_CACHE_DIR="${ROOT}/work/mesa-cache" \
    CUDA_CACHE_PATH="${ROOT}/work/cuda-cache" \
    PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    timeout --signal=TERM --kill-after=60s 7200s \
    xvfb-run -a -s '-screen 0 1024x768x24 -ac +extension GLX +render -noreset' \
    "${PYTHON}" "${CONTROL}/scripts/evaluate_exp0022_minecraft_parallel.py" \
    --experiment-id "${EXPERIMENT}" --runtime "${RUNTIME}" \
    --checkpoint "${CHECKPOINT}" --output "${ROOT}/evaluation" \
    --episodes 3 --agent-seed 10000 --episode-length 36000 --video-stride 4 \
    > "${ROOT}/stdout.log" 2>&1 &
EVAL_PID=$!
"${PYTHON}" "${CONTROL}/scripts/sample_exp0013_resources.py" \
  --watched-pid "${EVAL_PID}" --output-dir "${ROOT}" \
  --temp-root "${ROOT}/work/tmp" --interval 5 &
SAMPLER_PID=$!
set +e
wait "${EVAL_PID}"
eval_status=$?
set -e
stop_sampler
if (( eval_status != 0 )); then
  fail "${eval_status}" evaluation
fi
"${PYTHON}" "${CONTROL}/scripts/minecraft_temp_cleanup.py" \
  --experiment-id "${EXPERIMENT}" --temp-root "${ROOT}/work/tmp" \
  --output "${ROOT}/temp_cleanup_postprocess.json" --wait-seconds 30 \
  > "${ROOT}/temp_cleanup_postprocess_stdout.log" 2>&1 || fail $? temp_cleanup
printf \
  '{"experiment_id":"%s","completed_at":"%s","exit_code":0,"episodes":3,"total_wall_seconds":%d}\n' \
  "${EXPERIMENT}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  "$(( $(date +%s) - START_EPOCH ))" > "${ROOT}/.completed"
"${PYTHON}" "${CONTROL}/scripts/analyze_exp0023_parallel_eval.py" \
  --experiment-id "${EXPERIMENT}" --run-dir "${ROOT}" \
  --expected-checkpoint-agent-sha256 "${AGENT_SHA}" \
  --expected-checkpoint-step-sha256 "${STEP_SHA}" \
  --expected-episode-length 36000 --serial-wall-seconds 1972 \
  --serial-active-actions 55979 --minimum-wall-speedup 1.5 \
  --output "${ROOT}/analysis.json" > "${ROOT}/analysis_stdout.log" 2>&1 \
  || fail $? integrity
mapfile -t gpu_after < <(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d')
if (( ${#gpu_after[@]} )) || ps -eo comm= | awk '$1 == "java" || $1 == "Xvfb" {found=1} END {exit !found}'; then
  fail 28 cleanup
fi
sha256sum "${ROOT}/evaluation/evaluation.json" \
  "${ROOT}/evaluation/episode_000_preregistered_stride4.mp4" \
  "${ROOT}/evaluation/episode_000_first_frame.png" \
  "${ROOT}/analysis.json" > "${ROOT}/SHA256SUMS"
