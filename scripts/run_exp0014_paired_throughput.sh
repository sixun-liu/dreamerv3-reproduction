#!/usr/bin/env bash
set -euo pipefail

readonly EXPERIMENT=EXP-0014
readonly CONTROL=/root/autodl-tmp/Code/DreamerV3/dreamerv3-reproduction
readonly RUNTIME=/root/autodl-tmp/Code/DreamerV3/dreamerv3-runtime-2026-crossdomain
readonly WORKFLOW=/root/autodl-tmp/Tools/research-agent-kit
readonly PYTHON=/root/autodl-tmp/Envs/dv3-minecraft-2026/bin/python
readonly RUNTIME_COMMIT=5168475b7a4413f9575933b4580e7073caea2114
readonly RUN_ROOT=/root/autodl-tmp/Runs/EXP-0014__minecraft-paired-throughput__s31415__5040-env__20260812T061500Z
readonly ROOT=${RUN_ROOT}/envs1
readonly ENVS2=/root/autodl-tmp/Runs/EXP-0013__minecraft-throughput__s31415__5040-env__20260812T060000Z/envs2
readonly CONFIG=${CONTROL}/docs/reproduction/configs/exp0014_minecraft_envs1_s31415_5040_env.yaml
readonly MATRIX=${CONTROL}/docs/reproduction/configs/exp0014_minecraft_paired_throughput_matrix.yaml
readonly BASELINE=${CONTROL}/docs/reproduction/configs/exp0012_minecraft_smoke_s31415_4096_env.yaml
readonly STEPS=5040
readonly STARTED=${ROOT}.started
readonly COMPLETED=${ROOT}/.completed
readonly FAILED=${ROOT}/.failed
readonly CYCLE_COMPLETED=${RUN_ROOT}/.completed
readonly CYCLE_FAILED=${RUN_ROOT}/.failed
readonly STDOUT_LOG=${ROOT}/train_stdout.log
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
  local failure_signal=${FAILED}
  if [[ -f "${COMPLETED}" ]]; then
    failure_signal=${CYCLE_FAILED}
  fi
  printf \
    '{"experiment_id":"%s","arm":"envs1","failed_at":"%s","phase":"%s","exit_code":%d,"wall_seconds":%d}\n' \
    "${EXPERIMENT}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${phase}" \
    "${status}" "$(( $(date +%s) - START_EPOCH ))" > "${failure_signal}"
  exit "${status}"
}
trap stop_sampler EXIT

if [[ -e "${ROOT}" || -e "${STARTED}" ]]; then
  echo "Refusing duplicate EXP-0014 launch" >&2
  exit 20
fi
if [[ ! -f "${ENVS2}/.completed" || ! -f "${ENVS2}/integrity.json" ]]; then
  echo "Frozen EXP-0013 envs2 evidence is incomplete" >&2
  exit 21
fi
if [[ "$(sha256sum "${ENVS2}/.completed" | cut -d' ' -f1)" != \
      0f34fd25a31b035231552579b3bccb6590439dc77707cbf52136b3ff99663ebb || \
      "$(sha256sum "${ENVS2}/integrity.json" | cut -d' ' -f1)" != \
      91b23ec2acb1831e851cc6989dd5a367d17e236a21eb3772c9ef294445491854 || \
      "$(sha256sum "${ENVS2}/train/metrics.jsonl" | cut -d' ' -f1)" != \
      31616be2de73752f21718deb71d68119400b705a2fc0ff3a6f69a81deabfddff || \
      "$(sha256sum "${ENVS2}/resource_system.csv" | cut -d' ' -f1)" != \
      bd698bd7085dbb8d3b998e7add57966388939fed011aaeedbedd5e961ac1837b || \
      "$(sha256sum "${ENVS2}/resource_gpu.csv" | cut -d' ' -f1)" != \
      1c5152b5762bf3f2540ffb8306268d766b79a610a3a1309c0d6c482a2923a670 ]]; then
  echo "Frozen EXP-0013 evidence hash drift" >&2
  exit 22
fi
if [[ "$(git -C "${CONTROL}" status --porcelain)" ]]; then
  echo "Control repo must be clean" >&2
  exit 23
fi
if [[ "$(git -C "${RUNTIME}" status --porcelain)" || \
      "$(git -C "${RUNTIME}" rev-parse HEAD)" != "${RUNTIME_COMMIT}" ]]; then
  echo "Runtime provenance drift" >&2
  exit 24
fi
"${PYTHON}" "${CONTROL}/scripts/generate_exp0014_config.py" \
  --baseline "${BASELINE}" --run-root "${RUN_ROOT}" --output "${CONFIG}" \
  --check >/dev/null
mapfile -t gpu_pids < <(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d')
if (( ${#gpu_pids[@]} )); then
  echo "GPU is busy: ${gpu_pids[*]}" >&2
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
printf '{"experiment_id":"%s","arm":"envs1","started_at":"%s","pid":%d}\n' \
  "${EXPERIMENT}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$$" > "${STARTED}"
cp "${STARTED}" "${ROOT}/.started"
printf \
  '{"schema_version":1,"experiment_id":"%s","arm":"envs1","control_commit":"%s","runtime_commit":"%s","workflow_commit":"%s","config":"%s","config_sha256":"%s","matrix_sha256":"%s","seed":31415,"step_unit":"environment steps"}\n' \
  "${EXPERIMENT}" "$(git -C "${CONTROL}" rev-parse HEAD)" "${RUNTIME_COMMIT}" \
  "$(git -C "${WORKFLOW}" rev-parse HEAD)" "${CONFIG}" \
  "$(sha256sum "${CONFIG}" | cut -d' ' -f1)" \
  "$(sha256sum "${MATRIX}" | cut -d' ' -f1)" > "${ROOT}/.freeze"
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
    --run.envs 1 --run.debug False --run.steps "${STEPS}" \
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
  > "${ROOT}/temp_cleanup_postprocess_stdout.log" 2>&1 \
  || fail $? temp_cleanup
"${PYTHON}" "${CONTROL}/scripts/verify_exp0013_throughput.py" \
  --run-dir "${ROOT}" --frozen-config "${CONFIG}" \
  --expected-step "${STEPS}" --expected-envs 1 --driver-step-quantum 10 \
  --stdout-log "${STDOUT_LOG}" --resource-system "${ROOT}/resource_system.csv" \
  --temp-root "${ROOT}/work/tmp" --output "${ROOT}/integrity.json" \
  > "${ROOT}/integrity_stdout.log" 2>&1 || fail $? integrity
readonly WALL_SECONDS=$(( $(date +%s) - START_EPOCH ))
printf \
  '{"experiment_id":"%s","arm":"envs1","completed_at":"%s","exit_code":0,"requested_environment_steps":%d,"training_wall_seconds":%d,"total_wall_seconds":%d,"output_bytes":%d}\n' \
  "${EXPERIMENT}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${STEPS}" \
  "${TRAIN_WALL_SECONDS}" "${WALL_SECONDS}" "$(du -sb "${ROOT}" | awk '{print $1}')" \
  > "${COMPLETED}"
"${PYTHON}" "${CONTROL}/scripts/analyze_exp0014_paired_throughput.py" \
  --envs1-run "${ROOT}" --envs2-run "${ENVS2}" --steps "${STEPS}" \
  --output "${RUN_ROOT}/comparison.json" > "${RUN_ROOT}/comparison_stdout.log" 2>&1 \
  || fail $? paired_analysis
printf \
  '{"experiment_id":"%s","completed_at":"%s","exit_code":0,"envs1_arm_completed":true,"paired_analysis_completed":true,"total_wall_seconds":%d}\n' \
  "${EXPERIMENT}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  "$(( $(date +%s) - START_EPOCH ))" > "${CYCLE_COMPLETED}"
