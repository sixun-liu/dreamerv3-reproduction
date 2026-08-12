#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <completed-eval-run-dir>" >&2
  exit 2
fi
readonly EVAL_ROOT=$1
readonly CONTROL=/root/autodl-tmp/Code/DreamerV3/dreamerv3-reproduction
readonly PYTHON=/root/autodl-tmp/Envs/dv3-minecraft-2026/bin/python
readonly TRAIN_ROOT=/root/autodl-tmp/Runs/EXP-0012__minecraft-diamond__s000__100k-env__20260812T080000Z
readonly L0_JSON=/root/autodl-tmp/Runs/EXP-0012__minecraft-diamond__s31415__l0-32-step__20260812T080000Z/l0/minecraft_l0.json
readonly ARTIFACT_ROOT=/root/autodl-tmp/Artifacts/dreamerv3/EXP-0012
readonly TRAINING_OUTPUT=${ARTIFACT_ROOT}/training
readonly REVIEW_OUTPUT=/root/autodl-tmp/Artifacts/dreamerv3/review/EXP-0012-minecraft-diamond-reduced

if [[ ! -f "${TRAIN_ROOT}/.formal.completed" || \
      ! -f "${EVAL_ROOT}/.completed" || \
      ! -f "${EVAL_ROOT}/evaluation/evaluation.json" ]]; then
  echo "Formal training or evaluation completion is missing" >&2
  exit 20
fi
if [[ -e "${TRAINING_OUTPUT}" || -e "${REVIEW_OUTPUT}" ]]; then
  echo "Refusing to overwrite EXP-0012 postprocess outputs" >&2
  exit 21
fi

mkdir -p "${ARTIFACT_ROOT}"
"${PYTHON}" "${CONTROL}/scripts/analyze_exp0012_minecraft.py" \
  --run-dir "${TRAIN_ROOT}" \
  --inventory-schema "${L0_JSON}" \
  --output-dir "${TRAINING_OUTPUT}"
"${PYTHON}" "${CONTROL}/scripts/build_exp0012_review.py" \
  --training-analysis "${TRAINING_OUTPUT}/training_analysis.json" \
  --evaluation "${EVAL_ROOT}/evaluation/evaluation.json" \
  --output-dir "${REVIEW_OUTPUT}"
sha256sum "${TRAINING_OUTPUT}/training_analysis.json" \
  "${TRAINING_OUTPUT}/training_curve_and_milestones.png" \
  "${REVIEW_OUTPUT}/review_summary.json" \
  "${REVIEW_OUTPUT}/RESULT.md" > "${ARTIFACT_ROOT}/SHA256SUMS"
