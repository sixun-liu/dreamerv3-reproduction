#!/usr/bin/env bash
set -euo pipefail

readonly RUN=/root/autodl-tmp/Runs/EXP-0011__breakout__s000__100k-dec__20260811T201000Z
readonly CONTROL=/root/autodl-tmp/Code/DreamerV3/dreamerv3-reproduction
readonly RUNTIME=/root/autodl-tmp/Code/DreamerV3/dreamerv3-runtime-2026-crossdomain
readonly PYTHON=/root/autodl-tmp/Envs/dv3-atari-2026/bin/python
readonly ARTIFACT=/root/autodl-tmp/Artifacts/dreamerv3/EXP-0011
readonly REVIEW=/root/autodl-tmp/Artifacts/dreamerv3/review/EXP-0011-atari100k-breakout
readonly REFERENCE=${RUNTIME}/scores/atari100k-dreamerv3.json.gz
readonly STARTED=${RUN}/.postprocess.started
readonly COMPLETED=${RUN}/.postprocess.completed
readonly FAILED=${RUN}/.postprocess.failed

fail() {
  local status=$?
  printf '{"experiment_id":"EXP-0011","phase":"postprocess","failed_at":"%s","exit_code":%d}\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${status}" > "${FAILED}"
  exit "${status}"
}
trap fail ERR

if ! (set -o noclobber; printf '{"experiment_id":"EXP-0011","phase":"postprocess","started_at":"%s","pid":%d}\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$$" > "${STARTED}"); then
  echo "Refusing duplicate EXP-0011 postprocess" >&2
  exit 20
fi
while [[ ! -f "${RUN}/.formal.completed" ]]; do
  if [[ -f "${RUN}/.formal.failed" ]]; then
    echo "Formal training failed; postprocess will not run" >&2
    exit 21
  fi
  sleep 30
done
while nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d' | read -r; do
  sleep 5
done

"${PYTHON}" "${CONTROL}/scripts/analyze_exp0011_atari.py" \
  --stage formal \
  --run-dir "${RUN}" \
  --reference "${REFERENCE}" \
  --output-dir "${ARTIFACT}" \
  --review-dir "${REVIEW}" \
  > "${RUN}/analysis_formal_stdout.log" 2>&1

eval_tag="EXP-0011__breakout__eval-s10000-env20260812-10eps__$(date -u +%Y%m%dT%H%M%SZ)"
printf '%s\n' "${eval_tag}" > "${RUN}/postprocess_eval_tag.txt"
"${CONTROL}/scripts/run_exp0011_eval.sh" "${eval_tag}" \
  > "${RUN}/postprocess_eval_launcher.log" 2>&1
printf '{"experiment_id":"EXP-0011","phase":"postprocess","completed_at":"%s","evaluation_tag":"%s"}\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${eval_tag}" > "${COMPLETED}"
