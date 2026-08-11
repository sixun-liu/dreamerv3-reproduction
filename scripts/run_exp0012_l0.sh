#!/usr/bin/env bash
set -euo pipefail

readonly EXPERIMENT=EXP-0012
readonly CONTROL=/root/autodl-tmp/Code/DreamerV3/dreamerv3-reproduction
readonly RUNTIME=/root/autodl-tmp/Code/DreamerV3/dreamerv3-runtime-2026-crossdomain
readonly PYTHON=/root/autodl-tmp/Envs/dv3-minecraft-2026/bin/python
readonly WHEEL=/root/autodl-tmp/Staging/dv3-minecraft-wheel/minerl_mirror-0.4.4-cp311-cp311-linux_x86_64.whl
readonly TAG=EXP-0012__minecraft-diamond__s31415__l0-32-step__20260812T080000Z
readonly ROOT=/root/autodl-tmp/Runs/${TAG}
readonly STARTED=${ROOT}.started
readonly COMPLETED=${ROOT}.completed
readonly FAILED=${ROOT}.failed
readonly RUNTIME_COMMIT=5168475b7a4413f9575933b4580e7073caea2114
readonly WHEEL_SHA=b04e2cd21627e835d5b1566db1fe69ed1bff8abed1c804340ebd61c78a1dc631
readonly START_EPOCH=$(date +%s)

if [[ -e "${ROOT}" || -e "${STARTED}" ]]; then
  echo "Refusing duplicate EXP-0012 L0 launch" >&2
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
if [[ ! -x "${PYTHON}" || ! -s "${WHEEL}" ]]; then
  echo "Minecraft Python environment or wheel is missing" >&2
  exit 23
fi
if [[ "$(sha256sum "${WHEEL}" | cut -d' ' -f1)" != "${WHEEL_SHA}" ]]; then
  echo "MineRL wheel fingerprint drift" >&2
  exit 24
fi
if ! command -v java >/dev/null || ! command -v xvfb-run >/dev/null; then
  echo "Java or Xvfb is missing" >&2
  exit 25
fi
mapfile -t gpu_pids < <(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d')
if (( ${#gpu_pids[@]} )); then
  echo "GPU is busy: ${gpu_pids[*]}" >&2
  exit 26
fi

mkdir -p "${ROOT}"
printf '{"experiment_id":"%s","tag":"%s","started_at":"%s","pid":%d}\n' \
  "${EXPERIMENT}" "${TAG}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$$" > "${STARTED}"
cp "${STARTED}" "${ROOT}/.started"
printf \
  '{"schema_version":1,"experiment_id":"%s","purpose":"Minecraft L0","control_commit":"%s","runtime_commit":"%s","python":"%s","wheel_sha256":"%s","episode_steps_test_only":32}\n' \
  "${EXPERIMENT}" "$(git -C "${CONTROL}" rev-parse HEAD)" \
  "${RUNTIME_COMMIT}" "${PYTHON}" "${WHEEL_SHA}" > "${ROOT}.freeze"
cp "${ROOT}.freeze" "${ROOT}/.freeze"

fail() {
  local status=$?
  printf \
    '{"experiment_id":"%s","tag":"%s","failed_at":"%s","exit_code":%d,"wall_seconds":%d}\n' \
    "${EXPERIMENT}" "${TAG}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    "${status}" "$(( $(date +%s) - START_EPOCH ))" > "${FAILED}"
  cp "${FAILED}" "${ROOT}/.failed"
  exit "${status}"
}
trap fail ERR

{
  date -u +started_at=%Y-%m-%dT%H:%M:%SZ
  java -version
  "${PYTHON}" -m pip check
  df -B1 /root/autodl-tmp
  free -b
} > "${ROOT}/environment_before.txt" 2>&1

mkdir -p "${ROOT}/work"
cd "${ROOT}/work"
env PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  timeout --signal=TERM --kill-after=30s 900s \
  xvfb-run -a -s '-screen 0 1024x768x24 -ac +extension GLX +render -noreset' \
  "${PYTHON}" "${CONTROL}/scripts/check_exp0012_minecraft_env.py" \
  --runtime "${RUNTIME}" \
  --output "${ROOT}/l0" \
  --episode-steps 32 \
  > "${ROOT}/stdout.log" 2>&1

sha256sum "${ROOT}/l0/minecraft_l0.json" \
  "${ROOT}/l0/minecraft_l0_episode.mp4" \
  "${ROOT}/l0/minecraft_l0_first_frame.png" \
  "${ROOT}/l0/minecraft_l0_reset_frame.png" > "${ROOT}/SHA256SUMS"
printf \
  '{"experiment_id":"%s","tag":"%s","completed_at":"%s","exit_code":0,"wall_seconds":%d}\n' \
  "${EXPERIMENT}" "${TAG}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  "$(( $(date +%s) - START_EPOCH ))" > "${COMPLETED}"
cp "${COMPLETED}" "${ROOT}/.completed"
