#!/usr/bin/env bash
set -euo pipefail

if (( $# < 1 )); then
  echo "Usage: $0 <unique-run-tag> [seed ...]" >&2
  exit 2
fi

readonly TAG=$1
shift
if [[ "${TAG}" =~ ^EXP-0008__cheetah-run__showcase-eval-five-seed__[0-9]{8}T[0-9]{6}Z$ ]]; then
  readonly MODE=formal
elif [[ "${TAG}" =~ ^EXP-0008__cheetah-run__showcase-smoke-s([0-4])__[0-9]{8}T[0-9]{6}Z$ ]]; then
  readonly MODE=smoke
  readonly TAG_SEED=${BASH_REMATCH[1]}
else
  echo "Invalid showcase tag: ${TAG}" >&2
  exit 2
fi

if (( $# )); then
  SEEDS=("$@")
else
  SEEDS=(0 1 2 3 4)
fi
for seed in "${SEEDS[@]}"; do
  if [[ ! "${seed}" =~ ^[0-4]$ ]]; then
    echo "Invalid training seed: ${seed}" >&2
    exit 2
  fi
done
if [[ "${MODE}" == formal && "${SEEDS[*]}" != "0 1 2 3 4" ]]; then
  echo "Formal recording requires seeds in exact order: 0 1 2 3 4" >&2
  exit 2
fi
if [[ "${MODE}" == smoke && ( ${#SEEDS[@]} -ne 1 || "${SEEDS[0]}" != "${TAG_SEED}" ) ]]; then
  echo "Smoke tag seed and seed argument must match" >&2
  exit 2
fi

readonly EXPERIMENT=EXP-0008
readonly ROOT=/root/autodl-tmp/Runs/${TAG}
readonly CONTROL=/root/autodl-tmp/Code/DreamerV3/dreamerv3-reproduction
readonly RUNTIME=/root/autodl-tmp/Code/DreamerV3/dreamerv3-2411f7d
readonly MATRIX=/root/autodl-tmp/Runs/EXP-0008__cheetah-run__five-seed__500k-env__20260805T171000Z
readonly PYTHON=/root/autodl-tmp/Envs/dv3-2411/bin/python
readonly RECORDER=${CONTROL}/scripts/record_dreamerv3_checkpoint.py
readonly EVAL_SEED=10000
readonly EVAL_ENVS=1
readonly EVAL_DECISIONS=620

if [[ -e "${ROOT}" || -e "${ROOT}.started" || -e "${ROOT}.freeze" ]]; then
  echo "Refusing duplicate launch: ${ROOT} already exists" >&2
  exit 20
fi
mapfile -t gpu_pids < <(
  nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits |
    sed '/^[[:space:]]*$/d')
if (( ${#gpu_pids[@]} )); then
  echo "GPU is busy: ${gpu_pids[*]}" >&2
  exit 21
fi
available_kib=$(df --output=avail /root/autodl-tmp | tail -1)
if (( available_kib < 8 * 1024 * 1024 )); then
  echo "Data disk has less than 8 GiB available" >&2
  exit 22
fi

for seed in "${SEEDS[@]}"; do
  checkpoint=${MATRIX}/s$(printf '%03d' "${seed}")/train/checkpoint.ckpt
  if [[ ! -s "${checkpoint}" ]]; then
    echo "Missing checkpoint: ${checkpoint}" >&2
    exit 23
  fi
done

find /dev/shm -maxdepth 1 -type f \
  \( -name 'torch_*' -o -name 'sem.loky-*' -o -name 'cuda.shm.*' \) \
  -delete

mkdir -p "${ROOT}"
control_commit=$(git -C "${CONTROL}" rev-parse HEAD)
runtime_commit=$(git -C "${RUNTIME}" rev-parse HEAD)
workflow_commit=$(git -C /root/autodl-tmp/Tools/research-agent-kit rev-parse HEAD)
recorder_sha256=$(sha256sum "${RECORDER}" | cut -d' ' -f1)
seed_csv=$(IFS=,; printf '%s' "${SEEDS[*]}")
{
  printf '{\n'
  printf '  "experiment_id": "%s",\n' "${EXPERIMENT}"
  printf '  "purpose": "presentation-only terminal-checkpoint policy recording",\n'
  printf '  "mode": "%s",\n' "${MODE}"
  printf '  "created_at": "%s",\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf '  "control_commit": "%s",\n' "${control_commit}"
  printf '  "runtime_commit": "%s",\n' "${runtime_commit}"
  printf '  "workflow_commit": "%s",\n' "${workflow_commit}"
  printf '  "recorder_sha256": "%s",\n' "${recorder_sha256}"
  printf '  "task": "dmc_cheetah_run",\n'
  printf '  "train_seeds": [%s],\n' "${seed_csv}"
  printf '  "eval_seed": %d,\n' "${EVAL_SEED}"
  printf '  "eval_environment_seed_controlled": false,\n'
  printf '  "eval_envs": %d,\n' "${EVAL_ENVS}"
  printf '  "eval_agent_decisions": %d,\n' "${EVAL_DECISIONS}"
  printf '  "checkpoint_sha256": {\n'
  for index in "${!SEEDS[@]}"; do
    seed=${SEEDS[${index}]}
    checkpoint=${MATRIX}/s$(printf '%03d' "${seed}")/train/checkpoint.ckpt
    comma=,
    if (( index == ${#SEEDS[@]} - 1 )); then comma=; fi
    printf '    "s%03d": "%s"%s\n' "${seed}" \
      "$(sha256sum "${checkpoint}" | cut -d' ' -f1)" "${comma}"
  done
  printf '  }\n'
  printf '}\n'
} > "${ROOT}.freeze"
cp "${ROOT}.freeze" "${ROOT}/.freeze"

if ! (set -o noclobber; printf \
    '{"experiment_id":"%s","tag":"%s","started_at":"%s","pid":%d}\n' \
    "${EXPERIMENT}" "${TAG}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$$" \
    > "${ROOT}.started"); then
  echo "Refusing duplicate launch: ${ROOT}.started exists" >&2
  exit 24
fi
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

for seed in "${SEEDS[@]}"; do
  seed_name=s$(printf '%03d' "${seed}")
  checkpoint=${MATRIX}/${seed_name}/train/checkpoint.ckpt
  output=${ROOT}/${seed_name}
  mkdir -p "${output}"
  printf \
    '{"experiment_id":"%s","train_seed":%d,"eval_seed":%d,"started_at":"%s"}\n' \
    "${EXPERIMENT}" "${seed}" "${EVAL_SEED}" \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${output}/.started"
  cp "${ROOT}.freeze" "${output}/.freeze"

  cd "${CONTROL}"
  env PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
    "${PYTHON}" "${RECORDER}" \
    --runtime "${RUNTIME}" \
    --checkpoint "${checkpoint}" \
    --output "${output}/capture" \
    --task dmc_cheetah_run \
    --seed "${EVAL_SEED}" \
    --max-decisions "${EVAL_DECISIONS}" \
    --fps 20 \
    > "${output}/stdout.log" 2>&1 || fail $? "eval_${seed_name}"

  video=${output}/capture/policy.mp4
  if [[ -z "${video}" || ! -s "${video}" ]]; then
    echo "Evaluation completed without a policy video for ${seed_name}" >&2
    fail 25 "video_${seed_name}"
  fi
  if [[ ! -s "${output}/capture/episode.json" ]]; then
    echo "Evaluation completed without an episode score for ${seed_name}" >&2
    fail 26 "score_${seed_name}"
  fi
  printf '%s\n' "${video}" > "${output}/video_path.txt"
  sha256sum "${video}" > "${output}/video.sha256"
  printf \
    '{"experiment_id":"%s","train_seed":%d,"eval_seed":%d,"completed_at":"%s","exit_code":0}\n' \
    "${EXPERIMENT}" "${seed}" "${EVAL_SEED}" \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${output}/.completed"
done

printf \
  '{"experiment_id":"%s","tag":"%s","completed_at":"%s","exit_code":0,"seeds":%d}\n' \
  "${EXPERIMENT}" "${TAG}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  "${#SEEDS[@]}" > "${ROOT}.completed"
cp "${ROOT}.completed" "${ROOT}/.completed"
