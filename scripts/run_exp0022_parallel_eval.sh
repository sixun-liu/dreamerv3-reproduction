#!/usr/bin/env bash
set -euo pipefail

readonly EXPERIMENT=EXP-0022
readonly CONTROL=/root/autodl-tmp/Code/DreamerV3/dreamerv3-reproduction
readonly RUNTIME=/root/autodl-tmp/Code/DreamerV3/dreamerv3-exp0018-exact-stop
readonly RUNTIME_COMMIT=6723fc1620636cc8a52e0ddc0b7a9cbe946625b0
readonly PYTHON=/root/autodl-tmp/Envs/dv3-minecraft-2026/bin/python
readonly CHECKPOINT=/root/autodl-tmp/Runs/EXP-0020__minecraft-diamond__s000__100k-to-200k-env__20260812T100000Z/train/ckpt/20260812T182951F182513
readonly CHECKPOINT_AGENT_SHA256=2bc4067c82eabc241342a6bf83cb02d712f648eb2a9614ca214aee6d7255a01e
readonly CHECKPOINT_STEP_SHA256=d1250773c386047218484445fdab43ca63193d95993ccd89221576a4d4fa787b
readonly MATRIX=${CONTROL}/docs/reproduction/configs/exp0022_minecraft_parallel_eval_matrix.yaml
readonly ROOT=/root/autodl-tmp/Runs/EXP-0022__minecraft-runtime-milestones__s10000__3eps__20260812T200000Z
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
  printf \
    '{"experiment_id":"%s","failed_at":"%s","phase":"%s","exit_code":%d,"wall_seconds":%d}\n' \
    "${EXPERIMENT}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${phase}" \
    "${status}" "$(( $(date +%s) - START_EPOCH ))" > "${FAILED}"
  exit "${status}"
}
trap stop_sampler EXIT

if [[ -e "${ROOT}" || -e "${STARTED}" ]]; then
  echo "Refusing duplicate EXP-0022 launch" >&2
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
if [[ "$(sha256sum "${CHECKPOINT}/agent.pkl" | cut -d' ' -f1)" != \
      "${CHECKPOINT_AGENT_SHA256}" || \
      "$(sha256sum "${CHECKPOINT}/step.pkl" | cut -d' ' -f1)" != \
      "${CHECKPOINT_STEP_SHA256}" || ! -f "${CHECKPOINT}/done" ]]; then
  echo "Checkpoint provenance drift" >&2
  exit 23
fi
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

mkdir -p "${ROOT}"
printf '{"experiment_id":"%s","started_at":"%s","pid":%d}\n' \
  "${EXPERIMENT}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$$" > "${STARTED}"
cp "${STARTED}" "${ROOT}/.started"
printf \
  '{"schema_version":1,"experiment_id":"%s","control_commit":"%s","runtime_commit":"%s","matrix_sha256":"%s","checkpoint":"%s","checkpoint_agent_sha256":"%s","checkpoint_step_sha256":"%s","agent_seed":10000,"environment_count":3,"episodes_per_environment":1,"episode_length":36000,"video_worker":0,"video_stride":4}\n' \
  "${EXPERIMENT}" "$(git -C "${CONTROL}" rev-parse HEAD)" "${RUNTIME_COMMIT}" \
  "$(sha256sum "${MATRIX}" | cut -d' ' -f1)" "${CHECKPOINT}" \
  "${CHECKPOINT_AGENT_SHA256}" "${CHECKPOINT_STEP_SHA256}" > "${ROOT}/.freeze"
"${PYTHON}" -m pip freeze > "${ROOT}/python_environment.txt"

run_arm() {
  local name=$1
  local episode_length=$2
  local timeout_seconds=$3
  local arm=${ROOT}/${name}
  local arm_start
  arm_start=$(date +%s)
  mkdir -p "${arm}/work/tmp" "${arm}/work/malmo" "${arm}/work/jax-cache" \
    "${arm}/work/cache" "${arm}/work/mesa-cache" "${arm}/work/cuda-cache"
  find /dev/shm -maxdepth 1 -type f \
    \( -name 'torch_*' -o -name 'sem.loky-*' -o -name 'cuda.shm.*' \) -delete
  cd "${arm}/work"
  /usr/bin/time -v -o "${arm}/resource_time.txt" \
    env \
      TMPDIR="${arm}/work/tmp" TMP="${arm}/work/tmp" TEMP="${arm}/work/tmp" \
      MALMO_MINECRAFT_OUTPUT_LOGDIR="${arm}/work/malmo" \
      JAX_COMPILATION_CACHE_DIR="${arm}/work/jax-cache" \
      XDG_CACHE_HOME="${arm}/work/cache" MESA_SHADER_CACHE_DIR="${arm}/work/mesa-cache" \
      CUDA_CACHE_PATH="${arm}/work/cuda-cache" \
      PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      timeout --signal=TERM --kill-after=60s "${timeout_seconds}s" \
      xvfb-run -a -s '-screen 0 1024x768x24 -ac +extension GLX +render -noreset' \
      "${PYTHON}" "${CONTROL}/scripts/evaluate_exp0022_minecraft_parallel.py" \
      --experiment-id "${EXPERIMENT}" --runtime "${RUNTIME}" \
      --checkpoint "${CHECKPOINT}" --output "${arm}/evaluation" \
      --episodes 3 --agent-seed 10000 --episode-length "${episode_length}" \
      --video-stride 4 > "${arm}/stdout.log" 2>&1 &
  local eval_pid=$!
  "${PYTHON}" "${CONTROL}/scripts/sample_exp0013_resources.py" \
    --watched-pid "${eval_pid}" --output-dir "${arm}" \
    --temp-root "${arm}/work/tmp" --interval 5 &
  SAMPLER_PID=$!
  set +e
  wait "${eval_pid}"
  local eval_status=$?
  set -e
  stop_sampler
  if (( eval_status != 0 )); then
    fail "${eval_status}" "${name}_evaluation"
  fi
  "${PYTHON}" "${CONTROL}/scripts/minecraft_temp_cleanup.py" \
    --experiment-id "${EXPERIMENT}" --temp-root "${arm}/work/tmp" \
    --output "${arm}/temp_cleanup_postprocess.json" --wait-seconds 30 \
    > "${arm}/temp_cleanup_postprocess_stdout.log" 2>&1 \
    || fail $? "${name}_temp_cleanup"
  printf \
    '{"experiment_id":"%s","arm":"%s","completed_at":"%s","exit_code":0,"episode_length":%d,"total_wall_seconds":%d}\n' \
    "${EXPERIMENT}" "${name}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    "${episode_length}" "$(( $(date +%s) - arm_start ))" > "${arm}/.completed"
  "${PYTHON}" "${CONTROL}/scripts/analyze_exp0022_parallel_eval.py" \
    --run-dir "${arm}" \
    --expected-checkpoint-agent-sha256 "${CHECKPOINT_AGENT_SHA256}" \
    --expected-checkpoint-step-sha256 "${CHECKPOINT_STEP_SHA256}" \
    --expected-episode-length "${episode_length}" \
    --serial-wall-seconds 1972 --serial-active-actions 55979 \
    --minimum-wall-speedup 1.5 --output "${arm}/analysis.json" \
    > "${arm}/analysis_stdout.log" 2>&1 || fail $? "${name}_integrity"
  sha256sum "${arm}/evaluation/evaluation.json" \
    "${arm}/evaluation/episode_000_preregistered_stride4.mp4" \
    "${arm}/evaluation/episode_000_first_frame.png" \
    "${arm}/analysis.json" > "${arm}/SHA256SUMS"
  cd "${CONTROL}"
}

trap 'fail $? unexpected' ERR
run_arm smoke-128 128 1800
"${PYTHON}" - "${ROOT}/smoke-128/analysis.json" <<'PY'
import json
import sys
result = json.load(open(sys.argv[1], encoding="utf-8"))
assert result["integrity"]["passed"]
assert result["integrity"]["runtime_mapping_valid"]
assert result["integrity"]["cross_worker_mapping_consistent"]
assert result["integrity"]["reset_milestones_zero"]
PY
run_arm full-36000 36000 7200

mapfile -t gpu_after < <(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d')
if (( ${#gpu_after[@]} )) || ps -eo comm= | awk '$1 == "java" || $1 == "Xvfb" {found=1} END {exit !found}'; then
  fail 28 cleanup
fi
cp "${ROOT}/full-36000/analysis.json" "${ROOT}/analysis.json"
printf \
  '{"experiment_id":"%s","completed_at":"%s","exit_code":0,"total_wall_seconds":%d,"full_eval_wall_seconds":%s}\n' \
  "${EXPERIMENT}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  "$(( $(date +%s) - START_EPOCH ))" \
  "$("${PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["timing"]["parallel_total_wall_seconds"])' "${ROOT}/analysis.json")" \
  > "${ROOT}/.completed"
