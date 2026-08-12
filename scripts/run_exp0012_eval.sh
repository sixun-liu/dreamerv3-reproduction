#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <unique-eval-tag>" >&2
  exit 2
fi
readonly TAG=$1
if [[ ! "${TAG}" =~ ^EXP-0012__minecraft-diamond__eval-s10000-3eps__[0-9]{8}T[0-9]{6}Z$ ]]; then
  echo "Invalid eval tag: ${TAG}" >&2
  exit 2
fi

readonly EXPERIMENT=EXP-0012
readonly CONTROL=/root/autodl-tmp/Code/DreamerV3/dreamerv3-reproduction
readonly RUNTIME=/root/autodl-tmp/Code/DreamerV3/dreamerv3-runtime-2026-crossdomain
readonly RUNTIME_COMMIT=5168475b7a4413f9575933b4580e7073caea2114
readonly PYTHON=/root/autodl-tmp/Envs/dv3-minecraft-2026/bin/python
readonly TRAIN_ROOT=/root/autodl-tmp/Runs/EXP-0012__minecraft-diamond__s000__100k-env__20260812T080000Z
readonly CHECKPOINT_ROOT=${TRAIN_ROOT}/train/ckpt
readonly ROOT=/root/autodl-tmp/Runs/${TAG}

if [[ -e "${ROOT}" || -e "${ROOT}.started" ]]; then
  echo "Refusing duplicate EXP-0012 evaluation" >&2
  exit 20
fi
if [[ "$(git -C "${CONTROL}" status --porcelain)" ]]; then
  echo "Control repo must be clean before evaluation" >&2
  exit 21
fi
if [[ "$(git -C "${RUNTIME}" status --porcelain)" || \
      "$(git -C "${RUNTIME}" rev-parse HEAD)" != "${RUNTIME_COMMIT}" ]]; then
  echo "Runtime provenance drift" >&2
  exit 22
fi
if [[ ! -s "${CHECKPOINT_ROOT}/latest" || ! -f "${TRAIN_ROOT}/.formal.completed" ]]; then
  echo "Formal checkpoint is incomplete" >&2
  exit 23
fi
checkpoint_name=$(<"${CHECKPOINT_ROOT}/latest")
if [[ ! "${checkpoint_name}" =~ ^[A-Za-z0-9._-]+$ || \
      ! -s "${CHECKPOINT_ROOT}/${checkpoint_name}/agent.pkl" || \
      ! -s "${CHECKPOINT_ROOT}/${checkpoint_name}/step.pkl" || \
      ! -f "${CHECKPOINT_ROOT}/${checkpoint_name}/done" ]]; then
  echo "Formal directory checkpoint is incomplete" >&2
  exit 23
fi
readonly CHECKPOINT=${CHECKPOINT_ROOT}/${checkpoint_name}
mapfile -t gpu_pids < <(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d')
if (( ${#gpu_pids[@]} )); then
  echo "GPU is busy: ${gpu_pids[*]}" >&2
  exit 24
fi
if ! command -v java >/dev/null || ! command -v xvfb-run >/dev/null; then
  echo "Java or Xvfb is missing" >&2
  exit 25
fi

mkdir -p "${ROOT}"
printf \
  '{"experiment_id":"%s","purpose":"independent terminal checkpoint evaluation","control_commit":"%s","runtime_commit":"%s","checkpoint":"%s","checkpoint_agent_sha256":"%s","checkpoint_step_sha256":"%s","agent_seed":10000,"world_seed_controlled":false,"episodes":3,"episode_length":36000,"video_selection":"episode0","video_stride":4}\n' \
  "${EXPERIMENT}" "$(git -C "${CONTROL}" rev-parse HEAD)" \
  "${RUNTIME_COMMIT}" "${CHECKPOINT}" \
  "$(sha256sum "${CHECKPOINT}/agent.pkl" | cut -d' ' -f1)" \
  "$(sha256sum "${CHECKPOINT}/step.pkl" | cut -d' ' -f1)" > "${ROOT}.freeze"
cp "${ROOT}.freeze" "${ROOT}/.freeze"
printf '{"experiment_id":"%s","tag":"%s","started_at":"%s","pid":%d}\n' \
  "${EXPERIMENT}" "${TAG}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$$" > "${ROOT}.started"
cp "${ROOT}.started" "${ROOT}/.started"

fail() {
  local status=$?
  printf '{"experiment_id":"%s","tag":"%s","failed_at":"%s","exit_code":%d}\n' \
    "${EXPERIMENT}" "${TAG}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${status}" > "${ROOT}.failed"
  cp "${ROOT}.failed" "${ROOT}/.failed"
  exit "${status}"
}
trap fail ERR

/usr/bin/time -v -o "${ROOT}/resource_time.txt" \
  env PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  timeout --signal=TERM --kill-after=60s 21600s \
  xvfb-run -a -s '-screen 0 1024x768x24 -ac +extension GLX +render -noreset' \
  "${PYTHON}" "${CONTROL}/scripts/evaluate_exp0012_minecraft.py" \
  --runtime "${RUNTIME}" \
  --checkpoint "${CHECKPOINT}" \
  --output "${ROOT}/evaluation" \
  --episodes 3 \
  --agent-seed 10000 \
  --episode-length 36000 \
  --video-stride 4 \
  > "${ROOT}/stdout.log" 2>&1

"${PYTHON}" -c \
  'import json,sys; x=json.load(open(sys.argv[1])); assert x["episode_count"] == 3 and x["video_frame_count"] >= 2 and x["video_dynamic_adjacent_pairs"] > 0' \
  "${ROOT}/evaluation/evaluation.json"
sha256sum "${ROOT}/evaluation/evaluation.json" \
  "${ROOT}/evaluation/episode_000_preregistered_stride4.mp4" \
  "${ROOT}/evaluation/episode_000_first_frame.png" > "${ROOT}/SHA256SUMS"
printf '{"experiment_id":"%s","tag":"%s","completed_at":"%s","exit_code":0,"episodes":3}\n' \
  "${EXPERIMENT}" "${TAG}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${ROOT}.completed"
cp "${ROOT}.completed" "${ROOT}/.completed"
