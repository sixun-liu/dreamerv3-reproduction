#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <unique-run-tag>" >&2
  exit 2
fi

readonly TAG=$1
if [[ ! "${TAG}" =~ ^EXP-0006__walker-walk__showcase-eval-s001__[0-9]{8}T[0-9]{6}Z$ ]]; then
  echo "Invalid tag: ${TAG}" >&2
  exit 2
fi

readonly EXPERIMENT=EXP-0006
readonly ROOT=/root/autodl-tmp/runs/${TAG}
readonly CONTROL=/root/autodl-tmp/dreamerv3-reproduction
readonly RUNTIME=/root/autodl-tmp/dreamerv3-exp0006
readonly MATRIX=/root/autodl-tmp/runs/EXP-0006__walker-kl__matrix-2seed__20260721T142500Z
readonly PYTHON=/root/miniconda3/envs/dv3/bin/python
readonly EVAL_SEED=10000
readonly TRAIN_SEED=1
readonly EVAL_ENVS=1
readonly EVAL_AGENT_STEPS=520

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
if (( available_kib < 10 * 1024 * 1024 )); then
  echo "Data disk has less than 10 GiB available" >&2
  exit 22
fi

declare -A CHECKPOINTS
for arm in baseline e1 p4; do
  path_file=${MATRIX}/${arm}/s001/checkpoint_path.txt
  if [[ ! -s "${path_file}" ]]; then
    echo "Missing checkpoint pointer: ${path_file}" >&2
    exit 23
  fi
  CHECKPOINTS[${arm}]=$(<"${path_file}")
  if [[ ! -s "${CHECKPOINTS[${arm}]}/agent.pkl" ]]; then
    echo "Missing checkpoint agent: ${CHECKPOINTS[${arm}]}/agent.pkl" >&2
    exit 24
  fi
done

find /dev/shm -maxdepth 1 -type f \
  \( -name 'torch_*' -o -name 'sem.loky-*' -o -name 'cuda.shm.*' \) \
  -delete

mkdir -p "${ROOT}"
control_commit=$(git -C "${CONTROL}" rev-parse HEAD)
runtime_commit=$(git -C "${RUNTIME}" rev-parse HEAD)
workflow_commit=$(git -C /root/autodl-tmp/research-agent-kit rev-parse HEAD)
{
  printf '{\n'
  printf '  "experiment_id": "%s",\n' "${EXPERIMENT}"
  printf '  "purpose": "presentation-only fixed-condition policy recording",\n'
  printf '  "created_at": "%s",\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf '  "control_commit": "%s",\n' "${control_commit}"
  printf '  "runtime_commit": "%s",\n' "${runtime_commit}"
  printf '  "workflow_commit": "%s",\n' "${workflow_commit}"
  printf '  "train_seed": %d,\n' "${TRAIN_SEED}"
  printf '  "eval_seed": %d,\n' "${EVAL_SEED}"
  printf '  "eval_envs": %d,\n' "${EVAL_ENVS}"
  printf '  "eval_agent_steps": %d,\n' "${EVAL_AGENT_STEPS}"
  printf '  "task": "dmc_walker_walk",\n'
  printf '  "checkpoint_agent_sha256": {\n'
  printf '    "baseline": "%s",\n' "$(sha256sum "${CHECKPOINTS[baseline]}/agent.pkl" | cut -d' ' -f1)"
  printf '    "e1": "%s",\n' "$(sha256sum "${CHECKPOINTS[e1]}/agent.pkl" | cut -d' ' -f1)"
  printf '    "p4": "%s"\n' "$(sha256sum "${CHECKPOINTS[p4]}/agent.pkl" | cut -d' ' -f1)"
  printf '  },\n'
  printf '  "checkpoint_paths": {\n'
  printf '    "baseline": "%s",\n' "${CHECKPOINTS[baseline]}"
  printf '    "e1": "%s",\n' "${CHECKPOINTS[e1]}"
  printf '    "p4": "%s"\n' "${CHECKPOINTS[p4]}"
  printf '  }\n'
  printf '}\n'
} > "${ROOT}.freeze"
cp "${ROOT}.freeze" "${ROOT}/.freeze"

if ! (set -o noclobber; printf \
    '{"experiment_id":"%s","tag":"%s","started_at":"%s","pid":%d}\n' \
    "${EXPERIMENT}" "${TAG}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$$" \
    > "${ROOT}.started"); then
  echo "Refusing duplicate launch: ${ROOT}.started exists" >&2
  exit 25
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

for arm in baseline e1 p4; do
  output=${ROOT}/${arm}
  mkdir -p "${output}"
  printf \
    '{"experiment_id":"%s","arm":"%s","train_seed":%d,"eval_seed":%d,"started_at":"%s"}\n' \
    "${EXPERIMENT}" "${arm}" "${TRAIN_SEED}" "${EVAL_SEED}" \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${output}/.started"
  cp "${ROOT}.freeze" "${output}/.freeze"

  arm_overrides=()
  case "${arm}" in
    baseline) ;;
    e1)
      arm_overrides+=(--agent.dyn.rssm.free_nats 0.0)
      ;;
    p4)
      arm_overrides+=(
        --agent.dyn.rssm.free_nats 0.0
        --agent.loss_scales.rep 1.0)
      ;;
  esac

  cd "${RUNTIME}"
  env PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
    "${PYTHON}" dreamerv3/main.py \
    --logdir "${output}/eval" \
    --configs dmc_proprio size12m \
    --task dmc_walker_walk \
    --script eval_only \
    --seed "${EVAL_SEED}" \
    --env.dmc.use_seed True \
    --env.dmc.repeat 2 \
    --run.envs "${EVAL_ENVS}" \
    --run.steps "${EVAL_AGENT_STEPS}" \
    --run.log_every 1 \
    --run.from_checkpoint "${CHECKPOINTS[${arm}]}" \
    "${arm_overrides[@]}" \
    > "${output}/stdout.log" 2>&1 || fail $? "eval_${arm}"

  video=$(find "${output}/eval/scope" -path '*epstats-policy_log-image.mp4/*.mp4' \
    -type f -print -quit 2>/dev/null || true)
  if [[ -z "${video}" || ! -s "${video}" ]]; then
    echo "Evaluation completed without a policy video for ${arm}" >&2
    fail 26 "video_${arm}"
  fi
  printf '%s\n' "${video}" > "${output}/video_path.txt"
  sha256sum "${video}" > "${output}/video.sha256"
  printf \
    '{"experiment_id":"%s","arm":"%s","train_seed":%d,"eval_seed":%d,"completed_at":"%s","exit_code":0}\n' \
    "${EXPERIMENT}" "${arm}" "${TRAIN_SEED}" "${EVAL_SEED}" \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${output}/.completed"
done

printf \
  '{"experiment_id":"%s","tag":"%s","completed_at":"%s","exit_code":0,"arms":3}\n' \
  "${EXPERIMENT}" "${TAG}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  > "${ROOT}.completed"
cp "${ROOT}.completed" "${ROOT}/.completed"

