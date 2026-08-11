#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <unique-run-tag>" >&2
  exit 2
fi

readonly TAG=$1
if [[ ! "${TAG}" =~ ^EXP-0009__reacher-hard__showcase-eval-three-arm__[0-9]{8}T[0-9]{6}Z$ ]]; then
  echo "Invalid showcase tag: ${TAG}" >&2
  exit 2
fi

readonly EXPERIMENT=EXP-0009
readonly ROOT=/root/autodl-tmp/Runs/${TAG}
readonly CONTROL=/root/autodl-tmp/Code/DreamerV3/dreamerv3-reproduction
readonly RUNTIME=/root/autodl-tmp/Code/DreamerV3/dreamerv3-2411f7d
readonly MATRIX=/root/autodl-tmp/Runs/EXP-0009__reacher-hard__three-arm__s000__1m-env__20260806T034000Z
readonly PYTHON=/root/autodl-tmp/Envs/dv3-2411/bin/python
readonly RECORDER=${CONTROL}/scripts/record_dreamerv3_checkpoint.py
readonly BUILDER=${CONTROL}/scripts/build_exp0009_showcase.py
readonly ARTIFACTS=/root/autodl-tmp/Artifacts/dreamerv3/EXP-0009/showcase
readonly EVAL_SEED=10000
readonly EVAL_DECISIONS=620
readonly ARMS=(baseline no_reward_value no_reconstruction)

if [[ -e "${ROOT}" || -e "${ROOT}.started" || -e "${ROOT}.freeze" ]]; then
  echo "Refusing duplicate launch: ${ROOT} already exists" >&2
  exit 20
fi
if [[ -e "${ARTIFACTS}" ]]; then
  echo "Refusing to overwrite showcase artifacts: ${ARTIFACTS}" >&2
  exit 21
fi
mapfile -t gpu_pids < <(
  nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits |
    sed '/^[[:space:]]*$/d')
if (( ${#gpu_pids[@]} )); then
  echo "GPU is busy: ${gpu_pids[*]}" >&2
  exit 22
fi
available_kib=$(df --output=avail /root/autodl-tmp | tail -1)
if (( available_kib < 5 * 1024 * 1024 )); then
  echo "Data disk has less than 5 GiB available" >&2
  exit 23
fi
for arm in "${ARMS[@]}"; do
  checkpoint=${MATRIX}/${arm}/train/checkpoint.ckpt
  if [[ ! -s "${checkpoint}" ]]; then
    echo "Missing checkpoint: ${checkpoint}" >&2
    exit 24
  fi
done

find /dev/shm -maxdepth 1 -type f \
  \( -name 'torch_*' -o -name 'sem.loky-*' -o -name 'cuda.shm.*' \) \
  -delete
mkdir -p "${ROOT}"
{
  printf '{\n'
  printf '  "experiment_id": "%s",\n' "${EXPERIMENT}"
  printf '  "purpose": "presentation-only paired-initial-state policy recording",\n'
  printf '  "created_at": "%s",\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf '  "control_commit": "%s",\n' "$(git -C "${CONTROL}" rev-parse HEAD)"
  printf '  "runtime_commit": "%s",\n' "$(git -C "${RUNTIME}" rev-parse HEAD)"
  printf '  "recorder_sha256": "%s",\n' "$(sha256sum "${RECORDER}" | cut -d' ' -f1)"
  printf '  "builder_sha256": "%s",\n' "$(sha256sum "${BUILDER}" | cut -d' ' -f1)"
  printf '  "task": "dmc_reacher_hard",\n'
  printf '  "eval_seed": %d,\n' "${EVAL_SEED}"
  printf '  "eval_environment_seed_controlled": true,\n'
  printf '  "checkpoint_sha256": {\n'
  printf '    "baseline": "%s",\n' "$(sha256sum "${MATRIX}/baseline/train/checkpoint.ckpt" | cut -d' ' -f1)"
  printf '    "no_reward_value": "%s",\n' "$(sha256sum "${MATRIX}/no_reward_value/train/checkpoint.ckpt" | cut -d' ' -f1)"
  printf '    "no_reconstruction": "%s"\n' "$(sha256sum "${MATRIX}/no_reconstruction/train/checkpoint.ckpt" | cut -d' ' -f1)"
  printf '  }\n'
  printf '}\n'
} > "${ROOT}.freeze"
cp "${ROOT}.freeze" "${ROOT}/.freeze"
printf \
  '{"experiment_id":"%s","tag":"%s","started_at":"%s","pid":%d}\n' \
  "${EXPERIMENT}" "${TAG}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$$" \
  > "${ROOT}.started"
cp "${ROOT}.started" "${ROOT}/.started"

fail() {
  local status=$1
  local phase=$2
  printf \
    '{"experiment_id":"%s","tag":"%s","failed_at":"%s","phase":"%s","exit_code":%d}\n' \
    "${EXPERIMENT}" "${TAG}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    "${phase}" "${status}" > "${ROOT}.failed"
  cp "${ROOT}.failed" "${ROOT}/.failed"
  exit "${status}"
}

for arm in "${ARMS[@]}"; do
  output=${ROOT}/${arm}
  mkdir -p "${output}"
  cp "${ROOT}.freeze" "${output}/.freeze"
  checkpoint=${MATRIX}/${arm}/train/checkpoint.ckpt
  cd "${CONTROL}"
  env PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
    "${PYTHON}" "${RECORDER}" \
    --runtime "${RUNTIME}" \
    --checkpoint "${checkpoint}" \
    --output "${output}/capture" \
    --task dmc_reacher_hard \
    --seed "${EVAL_SEED}" \
    --use-env-seed \
    --max-decisions "${EVAL_DECISIONS}" \
    --fps 20 \
    > "${output}/stdout.log" 2>&1 || fail $? "eval_${arm}"
  test -s "${output}/capture/policy.mp4" || fail 25 "video_${arm}"
  test -s "${output}/capture/episode.json" || fail 26 "episode_${arm}"
  sha256sum "${output}/capture/policy.mp4" > "${output}/video.sha256"
done

"${PYTHON}" "${BUILDER}" --eval-root "${ROOT}" --output "${ARTIFACTS}" \
  > "${ROOT}/build_stdout.log" 2>&1 || fail $? build

printf \
  '{"experiment_id":"%s","tag":"%s","completed_at":"%s","exit_code":0,"arms":3,"artifacts":"%s"}\n' \
  "${EXPERIMENT}" "${TAG}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  "${ARTIFACTS}" > "${ROOT}.completed"
cp "${ROOT}.completed" "${ROOT}/.completed"
