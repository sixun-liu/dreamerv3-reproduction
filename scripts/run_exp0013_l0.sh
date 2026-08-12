#!/usr/bin/env bash
set -euo pipefail

readonly EXPERIMENT=EXP-0013
readonly CONTROL=/root/autodl-tmp/Code/DreamerV3/dreamerv3-reproduction
readonly RUNTIME=/root/autodl-tmp/Code/DreamerV3/dreamerv3-runtime-2026-crossdomain
readonly PYTHON=/root/autodl-tmp/Envs/dv3-minecraft-2026/bin/python
readonly RUNTIME_COMMIT=5168475b7a4413f9575933b4580e7073caea2114
readonly RUN_ROOT=/root/autodl-tmp/Runs/EXP-0013__minecraft-throughput__s31415__5040-env__20260812T060000Z
readonly ROOT=${RUN_ROOT}/l0-envs2-r1
readonly STARTED=${ROOT}.started
readonly COMPLETED=${ROOT}/.completed
readonly FAILED=${ROOT}/.failed
readonly START_EPOCH=$(date +%s)

fail() {
  local status=$?
  printf \
    '{"experiment_id":"%s","stage":"l0-envs2-r1","failed_at":"%s","exit_code":%d,"wall_seconds":%d}\n' \
    "${EXPERIMENT}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${status}" \
    "$(( $(date +%s) - START_EPOCH ))" > "${FAILED}"
  exit "${status}"
}
trap fail ERR

if [[ -e "${ROOT}" || -e "${STARTED}" ]]; then
  echo "Refusing duplicate EXP-0013 L0 retry" >&2
  exit 20
fi
if [[ ! -f "${RUN_ROOT}/l0-envs2/.failed" ]]; then
  echo "Original cleanup-gate failure is missing" >&2
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
if [[ ! -x "${PYTHON}" ]] || ! command -v java >/dev/null || ! command -v xvfb-run >/dev/null; then
  echo "Minecraft environment, Java, or Xvfb is missing" >&2
  exit 24
fi
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
printf '{"experiment_id":"%s","stage":"l0-envs2-r1","started_at":"%s","pid":%d}\n' \
  "${EXPERIMENT}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$$" > "${STARTED}"
cp "${STARTED}" "${ROOT}/.started"
printf \
  '{"schema_version":1,"experiment_id":"%s","control_commit":"%s","runtime_commit":"%s","envs":2,"temp_root":"%s"}\n' \
  "${EXPERIMENT}" "$(git -C "${CONTROL}" rev-parse HEAD)" "${RUNTIME_COMMIT}" \
  "${ROOT}/work/tmp" > "${ROOT}/.freeze"

find /dev/shm -maxdepth 1 -type f \
  \( -name 'torch_*' -o -name 'sem.loky-*' -o -name 'cuda.shm.*' \) -delete
cd "${ROOT}/work"
env \
  TMPDIR="${ROOT}/work/tmp" TMP="${ROOT}/work/tmp" TEMP="${ROOT}/work/tmp" \
  MALMO_MINECRAFT_OUTPUT_LOGDIR="${ROOT}/work/malmo" \
  JAX_COMPILATION_CACHE_DIR="${ROOT}/work/jax-cache" \
  XDG_CACHE_HOME="${ROOT}/work/cache" MESA_SHADER_CACHE_DIR="${ROOT}/work/mesa-cache" \
  CUDA_CACHE_PATH="${ROOT}/work/cuda-cache" \
  PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  timeout --signal=TERM --kill-after=30s 900s \
  xvfb-run -a -s '-screen 0 1024x768x24 -ac +extension GLX +render -noreset' \
  "${PYTHON}" "${CONTROL}/scripts/check_exp0013_multienv.py" \
  --runtime "${RUNTIME}" --output "${ROOT}/l0" --envs 2 --steps-per-env 4 \
  > "${ROOT}/stdout.log" 2>&1

"${PYTHON}" - "${ROOT}/l0/minecraft_multienv_l0.json" \
  "${ROOT}/l0/temp_dirs_after_close.json" <<'PY'
import json
import sys
summary = json.load(open(sys.argv[1], encoding="utf-8"))
cleanup = json.load(open(sys.argv[2], encoding="utf-8"))
if not summary["passed"]:
    raise SystemExit((summary.get("passed"), cleanup))
PY
"${PYTHON}" "${CONTROL}/scripts/cleanup_exp0013_temp.py" \
  --temp-root "${ROOT}/work/tmp" \
  --output "${ROOT}/l0/temp_cleanup_postprocess.json" --wait-seconds 30 \
  > "${ROOT}/l0/temp_cleanup_postprocess_stdout.log" 2>&1
printf \
  '{"experiment_id":"%s","stage":"l0-envs2-r1","completed_at":"%s","exit_code":0,"wall_seconds":%d}\n' \
  "${EXPERIMENT}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  "$(( $(date +%s) - START_EPOCH ))" > "${COMPLETED}"
