#!/usr/bin/env bash
set -euo pipefail

readonly EXPERIMENT=EXP-0011
readonly CONTROL=/root/autodl-tmp/Code/DreamerV3/dreamerv3-reproduction
readonly RUNTIME=/root/autodl-tmp/Code/DreamerV3/dreamerv3-runtime-2026-crossdomain
readonly PYTHON=/root/autodl-tmp/Envs/dv3-atari-2026/bin/python
readonly ROMDIR=/root/autodl-tmp/ThirdParty/atari-roms
readonly TAG=EXP-0011__breakout__s31415__ale-l0__20260811T201000Z
readonly ROOT=/root/autodl-tmp/Runs/${TAG}
readonly STARTED=${ROOT}.started
readonly COMPLETED=${ROOT}.completed
readonly FAILED=${ROOT}.failed

if [[ -e "${ROOT}" || -e "${STARTED}" ]]; then
  echo "Refusing duplicate EXP-0011 L0 launch" >&2
  exit 20
fi
if [[ "$(git -C "${CONTROL}" status --porcelain)" ]]; then
  echo "Control repo must be clean" >&2
  exit 21
fi
if [[ "$(git -C "${RUNTIME}" status --porcelain)" ]]; then
  echo "Runtime repo must be clean" >&2
  exit 22
fi
if [[ "$(git -C "${RUNTIME}" rev-parse HEAD)" != "5168475b7a4413f9575933b4580e7073caea2114" ]]; then
  echo "Runtime commit drift" >&2
  exit 23
fi
if [[ ! -x "${PYTHON}" || ! -s "${ROMDIR}/breakout.bin" ]]; then
  echo "Atari environment or ROM is missing" >&2
  exit 24
fi

mkdir -p "${ROOT}"
printf '{"experiment_id":"%s","tag":"%s","started_at":"%s","pid":%d}\n' \
  "${EXPERIMENT}" "${TAG}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$$" > "${STARTED}"
cp "${STARTED}" "${ROOT}/.started"
printf \
  '{"experiment_id":"%s","purpose":"ALE L0","control_commit":"%s","runtime_commit":"%s","rom_md5":"f34f08e5eb96e500e851a80be3277a56","seed":31415}\n' \
  "${EXPERIMENT}" "$(git -C "${CONTROL}" rev-parse HEAD)" \
  "$(git -C "${RUNTIME}" rev-parse HEAD)" > "${ROOT}.freeze"
cp "${ROOT}.freeze" "${ROOT}/.freeze"

fail() {
  local status=$?
  printf '{"experiment_id":"%s","tag":"%s","failed_at":"%s","exit_code":%d}\n' \
    "${EXPERIMENT}" "${TAG}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${status}" > "${FAILED}"
  cp "${FAILED}" "${ROOT}/.failed"
  exit "${status}"
}
trap fail ERR

env ALE_ROM_PATH="${ROMDIR}" PYTHONUNBUFFERED=1 \
  "${PYTHON}" "${CONTROL}/scripts/check_exp0011_atari_env.py" \
  --runtime "${RUNTIME}" \
  --rom-dir "${ROMDIR}" \
  --output "${ROOT}/l0" \
  --seed 31415 \
  > "${ROOT}/stdout.log" 2>&1

sha256sum "${ROOT}/l0/ale_l0.json" "${ROOT}/l0/ale_l0_episode.mp4" \
  "${ROOT}/l0/ale_l0_first_frame.png" > "${ROOT}/SHA256SUMS"
printf '{"experiment_id":"%s","tag":"%s","completed_at":"%s","exit_code":0}\n' \
  "${EXPERIMENT}" "${TAG}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${COMPLETED}"
cp "${COMPLETED}" "${ROOT}/.completed"

