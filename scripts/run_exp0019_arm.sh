#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || ! "$1" =~ ^(4|5)$ ]]; then
  echo "Usage: $0 <4|5>" >&2
  exit 2
fi

readonly ENVS=$1
readonly EXPERIMENT=EXP-0019
readonly CONTROL=/root/autodl-tmp/Code/DreamerV3/dreamerv3-reproduction
readonly RUNTIME=/root/autodl-tmp/Code/DreamerV3/dreamerv3-exp0018-exact-stop
readonly WORKFLOW=/root/autodl-tmp/Tools/research-agent-kit
readonly PYTHON=/root/autodl-tmp/Envs/dv3-minecraft-2026/bin/python
readonly RUNTIME_COMMIT=6723fc1620636cc8a52e0ddc0b7a9cbe946625b0
readonly SOURCE=/root/autodl-tmp/Runs/EXP-0012__minecraft-diamond__s000__100k-env__20260812T080000Z/train
readonly RUN_ROOT=/root/autodl-tmp/Runs/EXP-0019__minecraft-local-scaling__s000__100k-to-105040-env__20260812T093000Z
readonly ROOT=${RUN_ROOT}/envs${ENVS}
readonly CONFIG=${CONTROL}/docs/reproduction/configs/exp0019_minecraft_recovery_envs${ENVS}_s000_105040_env.yaml
readonly MATRIX=${CONTROL}/docs/reproduction/configs/exp0019_minecraft_local_scaling_matrix.yaml
readonly BASELINE_CONFIG=${CONTROL}/docs/reproduction/configs/exp0012_minecraft_s000_100k_env.yaml
readonly SOURCE_STEP=100000
readonly FINAL_STEP=105040
readonly SOURCE_CHECKPOINT_TREE_SHA256=768e088455502cadbee875022f7754c7703b5df6708aa7f9fed4df4c70775fe0
readonly SOURCE_REPLAY_TREE_SHA256=f103990afb0a5f484aa965a03df110d0a0a8ae9c12f2a9c823757bac95127fc2
readonly STARTED=${ROOT}.started
readonly COMPLETED=${ROOT}/.completed
readonly FAILED=${ROOT}/.failed
readonly STDOUT_LOG=${ROOT}/train_stdout.log
readonly CLONE_MANIFEST=${ROOT}/recovery_clone_manifest.json
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
    '{"experiment_id":"%s","arm":"envs%s","failed_at":"%s","phase":"%s","exit_code":%d,"wall_seconds":%d}\n' \
    "${EXPERIMENT}" "${ENVS}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${phase}" \
    "${status}" "$(( $(date +%s) - START_EPOCH ))" > "${FAILED}"
  exit "${status}"
}
trap stop_sampler EXIT

if [[ -e "${ROOT}" || -e "${STARTED}" ]]; then
  echo "Refusing duplicate EXP-0019 envs${ENVS} launch" >&2
  exit 20
fi
if [[ ! -f "${SOURCE}/../.completed" || ! -f "${SOURCE}/ckpt/latest" ]]; then
  echo "EXP-0012 recovery source is incomplete" >&2
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
"${PYTHON}" "${CONTROL}/scripts/generate_exp0019_configs.py" \
  --baseline "${BASELINE_CONFIG}" --run-root "${RUN_ROOT}" --envs "${ENVS}" \
  --output "${CONFIG}" --check >/dev/null
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
printf '{"experiment_id":"%s","arm":"envs%s","started_at":"%s","pid":%d}\n' \
  "${EXPERIMENT}" "${ENVS}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$$" > "${STARTED}"
cp "${STARTED}" "${ROOT}/.started"
printf \
  '{"schema_version":1,"experiment_id":"%s","arm":"envs%s","control_commit":"%s","runtime_commit":"%s","workflow_commit":"%s","config_sha256":"%s","matrix_sha256":"%s","seed":0,"source_step":%d,"absolute_final_step":%d,"cache_mode":"independent_cold"}\n' \
  "${EXPERIMENT}" "${ENVS}" "$(git -C "${CONTROL}" rev-parse HEAD)" "${RUNTIME_COMMIT}" \
  "$(git -C "${WORKFLOW}" rev-parse HEAD)" "$(sha256sum "${CONFIG}" | cut -d' ' -f1)" \
  "$(sha256sum "${MATRIX}" | cut -d' ' -f1)" "${SOURCE_STEP}" "${FINAL_STEP}" \
  > "${ROOT}/.freeze"
"${PYTHON}" -m pip freeze > "${ROOT}/python_environment.txt"
"${PYTHON}" "${CONTROL}/scripts/prepare_exp0015_recovery.py" \
  --source-train "${SOURCE}" --target-train "${ROOT}/train" \
  --expected-step "${SOURCE_STEP}" \
  --expected-checkpoint-sha256 "${SOURCE_CHECKPOINT_TREE_SHA256}" \
  --expected-replay-sha256 "${SOURCE_REPLAY_TREE_SHA256}" \
  --output "${CLONE_MANIFEST}" > "${ROOT}/recovery_clone_stdout.log" 2>&1 \
  || fail $? clone

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
    timeout --signal=TERM --kill-after=60s 2400s \
    xvfb-run -a -s '-screen 0 1024x768x24 -ac +extension GLX +render -noreset' \
    "${PYTHON}" "${RUNTIME}/dreamerv3/main.py" \
    --logdir "${ROOT}/train" --configs minecraft size50m \
    --task minecraft_diamond --script train --seed 0 \
    --run.envs "${ENVS}" --run.debug False --run.steps "${FINAL_STEP}" \
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
"${PYTHON}" "${CONTROL}/scripts/minecraft_temp_cleanup.py" \
  --experiment-id "${EXPERIMENT}" --temp-root "${ROOT}/work/tmp" \
  --output "${ROOT}/temp_cleanup_postprocess.json" --wait-seconds 30 \
  > "${ROOT}/temp_cleanup_postprocess_stdout.log" 2>&1 || fail $? temp_cleanup
"${PYTHON}" "${CONTROL}/scripts/verify_exp0015_recovery.py" \
  --experiment-id "${EXPERIMENT}" --expected-envs "${ENVS}" \
  --run-dir "${ROOT}" --source-train "${SOURCE}" --source-config "${BASELINE_CONFIG}" \
  --clone-manifest "${CLONE_MANIFEST}" --frozen-config "${CONFIG}" \
  --source-step "${SOURCE_STEP}" --final-step "${FINAL_STEP}" \
  --stdout-log "${STDOUT_LOG}" --resource-system "${ROOT}/resource_system.csv" \
  --max-cgroup-memory-bytes 77309411328 \
  --temp-root "${ROOT}/work/tmp" --output "${ROOT}/integrity.json" \
  > "${ROOT}/integrity_stdout.log" 2>&1 || fail $? integrity
readonly WALL_SECONDS=$(( $(date +%s) - START_EPOCH ))
printf \
  '{"experiment_id":"%s","arm":"envs%s","completed_at":"%s","exit_code":0,"source_step":%d,"absolute_final_step":%d,"added_environment_steps":5040,"training_wall_seconds":%d,"total_wall_seconds":%d,"output_bytes":%d}\n' \
  "${EXPERIMENT}" "${ENVS}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${SOURCE_STEP}" \
  "${FINAL_STEP}" "${TRAIN_WALL_SECONDS}" "${WALL_SECONDS}" \
  "$(du -sb "${ROOT}" | awk '{print $1}')" > "${COMPLETED}"
