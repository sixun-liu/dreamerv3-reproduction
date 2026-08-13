#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/quickstart.sh <dmc-vision|breakout|minecraft> [check|l0|smoke|pilot|formal] [--dry-run]

Examples:
  scripts/quickstart.sh dmc-vision check
  scripts/quickstart.sh dmc-vision smoke
  scripts/quickstart.sh breakout l0
  scripts/quickstart.sh breakout smoke
  scripts/quickstart.sh minecraft smoke --dry-run

Environment overrides:
  DREAMERV3_RUNTIME_2024       2024 author runtime checkout
  DREAMERV3_RUNTIME_2026       2026 author runtime checkout
  DREAMERV3_PYTHON_DMC         Python executable for DMC Vision
  DREAMERV3_PYTHON_ATARI       Python executable for Atari
  DREAMERV3_PYTHON_MINECRAFT   Python executable for Minecraft
  DREAMERV3_ROM_DIR            Directory containing breakout.bin
  DREAMERV3_OUTPUT_ROOT        Root directory for new runs
EOF
}

if (( $# < 1 )); then
  usage >&2
  exit 2
fi

TASK=$1
shift
STAGE=${1:-smoke}
if (( $# )); then
  shift
fi
DRY_RUN=false
while (( $# )); do
  case "$1" in
    --dry-run) DRY_RUN=true ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

CONTROL=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
WORKSPACE=$(cd "${CONTROL}/.." && pwd)
if [[ -n "${DREAMERV3_OUTPUT_ROOT:-}" ]]; then
  OUTPUT_ROOT=${DREAMERV3_OUTPUT_ROOT}
elif [[ -d /root/autodl-tmp/Runs ]]; then
  OUTPUT_ROOT=/root/autodl-tmp/Runs
else
  OUTPUT_ROOT=${CONTROL}/outputs
fi
RUNTIME_2024=${DREAMERV3_RUNTIME_2024:-${WORKSPACE}/dreamerv3-runtime-2411-crossdomain}
RUNTIME_2026=${DREAMERV3_RUNTIME_2026:-${WORKSPACE}/dreamerv3-runtime-2026-crossdomain}
PYTHON_DMC=${DREAMERV3_PYTHON_DMC:-/root/autodl-tmp/Envs/dv3-2411/bin/python}
PYTHON_ATARI=${DREAMERV3_PYTHON_ATARI:-/root/autodl-tmp/Envs/dv3-atari-2026/bin/python}
PYTHON_MINECRAFT=${DREAMERV3_PYTHON_MINECRAFT:-/root/autodl-tmp/Envs/dv3-minecraft-2026/bin/python}
if [[ -n "${DREAMERV3_ROM_DIR:-}" ]]; then
  ROM_DIR=${DREAMERV3_ROM_DIR}
elif [[ -d /root/autodl-tmp/ThirdParty/atari-roms ]]; then
  ROM_DIR=/root/autodl-tmp/ThirdParty/atari-roms
else
  ROM_DIR=${WORKSPACE}/ThirdParty/atari-roms
fi

RUNTIME_2024_COMMIT=6642b941f578cd72147bc2be3c3343d5bc72931c
RUNTIME_2026_COMMIT=5168475b7a4413f9575933b4580e7073caea2114
BREAKOUT_MD5=f34f08e5eb96e500e851a80be3277a56

case "${TASK}" in
  dmc-vision)
    RUNTIME=${RUNTIME_2024}
    PYTHON=${PYTHON_DMC}
    EXPECTED_COMMIT=${RUNTIME_2024_COMMIT}
    STEP_UNIT=agent_decisions
    ENV_STEP_MULTIPLIER=2
    case "${STAGE}" in
      check) STEPS=0; SEED=31415; TRAIN_RATIO=512 ;;
      smoke) STEPS=8192; SEED=31415; TRAIN_RATIO=512 ;;
      pilot) STEPS=50000; SEED=0; TRAIN_RATIO=512 ;;
      formal) STEPS=500000; SEED=0; TRAIN_RATIO=512 ;;
      *) echo "DMC Vision supports check, smoke, pilot, or formal" >&2; exit 2 ;;
    esac
    ;;
  breakout)
    RUNTIME=${RUNTIME_2026}
    PYTHON=${PYTHON_ATARI}
    EXPECTED_COMMIT=${RUNTIME_2026_COMMIT}
    STEP_UNIT=agent_decisions
    ENV_STEP_MULTIPLIER=4
    case "${STAGE}" in
      check|l0) STEPS=0; SEED=31415; TRAIN_RATIO=256 ;;
      smoke) STEPS=4090; SEED=31415; TRAIN_RATIO=256 ;;
      formal) STEPS=100000; SEED=0; TRAIN_RATIO=256 ;;
      *) echo "Breakout supports check, l0, smoke, or formal" >&2; exit 2 ;;
    esac
    ;;
  minecraft)
    RUNTIME=${RUNTIME_2026}
    PYTHON=${PYTHON_MINECRAFT}
    EXPECTED_COMMIT=${RUNTIME_2026_COMMIT}
    STEP_UNIT=environment_steps
    ENV_STEP_MULTIPLIER=1
    case "${STAGE}" in
      check|l0) STEPS=0; SEED=31415; TRAIN_RATIO=32 ;;
      smoke) STEPS=4096; SEED=31415; TRAIN_RATIO=32 ;;
      formal) STEPS=100000; SEED=0; TRAIN_RATIO=32 ;;
      *) echo "Minecraft supports check, l0, smoke, or formal" >&2; exit 2 ;;
    esac
    ;;
  -h|--help) usage; exit 0 ;;
  *) echo "Unknown task: ${TASK}" >&2; usage >&2; exit 2 ;;
esac

require_file() {
  [[ -f "$1" ]] || { echo "Missing file: $1" >&2; return 1; }
}

require_command() {
  command -v "$1" >/dev/null || { echo "Missing command: $1" >&2; return 1; }
}

check_python_imports() {
  local modules=$1
  env PYTHONPATH="${RUNTIME}${PYTHONPATH:+:${PYTHONPATH}}" \
    "${PYTHON}" - "${modules}" <<'PY'
import importlib
import sys

missing = []
for name in sys.argv[1].split(","):
    try:
        importlib.import_module(name)
    except Exception as exc:
        missing.append(f"{name}: {exc}")
if missing:
    raise SystemExit("Missing Python imports:\n  " + "\n  ".join(missing))
PY
}

preflight() {
  require_command git
  require_command nvidia-smi
  [[ -z "$(git -C "${CONTROL}" status --porcelain)" ]] || {
    echo "Control checkout must be clean: ${CONTROL}" >&2
    return 1
  }
  require_file "${RUNTIME}/dreamerv3/main.py"
  [[ -x "${PYTHON}" ]] || { echo "Python is not executable: ${PYTHON}" >&2; return 1; }
  [[ "$(git -C "${RUNTIME}" rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]] || {
    echo "Runtime commit mismatch in ${RUNTIME}" >&2
    echo "Expected ${EXPECTED_COMMIT}" >&2
    return 1
  }
  [[ -z "$(git -C "${RUNTIME}" status --porcelain)" ]] || {
    echo "Runtime checkout must be clean: ${RUNTIME}" >&2
    return 1
  }
  mapfile -t gpu_pids < <(
    nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null |
      sed '/^[[:space:]]*$/d')
  if (( ${#gpu_pids[@]} )); then
    echo "GPU is busy: ${gpu_pids[*]}" >&2
    return 1
  fi
  case "${TASK}" in
    dmc-vision)
      check_python_imports "jax,dm_control,embodied"
      ;;
    breakout)
      require_file "${ROM_DIR}/breakout.bin"
      [[ "$(md5sum "${ROM_DIR}/breakout.bin" | cut -d' ' -f1)" == "${BREAKOUT_MD5}" ]] || {
        echo "Breakout ROM fingerprint mismatch" >&2
        return 1
      }
      check_python_imports "jax,ale_py,av,PIL,elements,embodied"
      ;;
    minecraft)
      require_command java
      require_command xvfb-run
      require_command timeout
      if ! java -version 2>&1 | head -1 | grep -Eq 'version "1\.8|version "8'; then
        echo "Minecraft path requires Java 8" >&2
        return 1
      fi
      check_python_imports "jax,minerl,av,PIL,elements,embodied"
      ;;
  esac
  mkdir -p "${OUTPUT_ROOT}"
  local available_kib
  available_kib=$(df --output=avail "${OUTPUT_ROOT}" | tail -1)
  if (( available_kib < 5 * 1024 * 1024 )); then
    echo "Output filesystem has less than 5 GiB free" >&2
    return 1
  fi
  printf 'Preflight passed\n  task: %s\n  runtime: %s\n  python: %s\n  output: %s\n' \
    "${TASK}" "${RUNTIME}" "${PYTHON}" "${OUTPUT_ROOT}"
}

if [[ "${STAGE}" == "check" ]]; then
  preflight
  exit 0
fi

TIMESTAMP=$(date -u +%Y%m%dT%H%M%SZ)
RUN_DIR=${OUTPUT_ROOT}/quickstart-${TASK}-${STAGE}-${TIMESTAMP}
TRAIN_DIR=${RUN_DIR}/train
STDOUT_LOG=${RUN_DIR}/train_stdout.log
VERIFY=${CONTROL}/scripts/verify_quickstart.py

case "${TASK}:${STAGE}" in
  breakout:l0)
    COMMAND=(env ALE_ROM_PATH="${ROM_DIR}" PYTHONUNBUFFERED=1
      "${PYTHON}" "${CONTROL}/scripts/check_exp0011_atari_env.py"
      --runtime "${RUNTIME}" --rom-dir "${ROM_DIR}" --output "${RUN_DIR}/l0"
      --seed "${SEED}")
    ;;
  minecraft:l0)
    COMMAND=(env PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
      timeout --signal=TERM --kill-after=30s 900s
      xvfb-run -a -s "-screen 0 1024x768x24 -ac +extension GLX +render -noreset"
      "${PYTHON}" "${CONTROL}/scripts/check_exp0012_minecraft_env.py"
      --runtime "${RUNTIME}" --output "${RUN_DIR}/l0" --episode-steps 32)
    ;;
  dmc-vision:*)
    COMMAND=(env PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
      MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONPATH="${RUNTIME}${PYTHONPATH:+:${PYTHONPATH}}"
      "${PYTHON}" "${RUNTIME}/dreamerv3/main.py"
      --logdir "${TRAIN_DIR}" --configs dmc_vision size12m
      --task dmc_walker_walk --run.script train --seed "${SEED}"
      --tensorboard False --env.dmc.repeat 2 --run.num_envs 16
      --run.steps "${STEPS}" --run.train_ratio "${TRAIN_RATIO}"
      --run.log_every 120 --run.save_every 600 --run.save_at_end True)
    ;;
  breakout:*)
    LOG_EVERY=120
    [[ "${STAGE}" == "smoke" ]] && LOG_EVERY=15
    COMMAND=(env ALE_ROM_PATH="${ROM_DIR}" PYTHONUNBUFFERED=1
      OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH="${RUNTIME}${PYTHONPATH:+:${PYTHONPATH}}"
      "${PYTHON}" "${RUNTIME}/dreamerv3/main.py"
      --logdir "${TRAIN_DIR}" --configs atari100k size50m
      --task atari100k_breakout --script train --seed "${SEED}"
      --run.envs 1 --run.steps "${STEPS}" --run.train_ratio "${TRAIN_RATIO}"
      --run.log_every "${LOG_EVERY}" --run.save_at_end True)
    ;;
  minecraft:*)
    LOG_EVERY=120
    REPORT_EVERY=300
    SAVE_EVERY=900
    if [[ "${STAGE}" == "smoke" ]]; then
      LOG_EVERY=30
      REPORT_EVERY=120
      SAVE_EVERY=300
    fi
    COMMAND=(env PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
      PYTHONPATH="${RUNTIME}${PYTHONPATH:+:${PYTHONPATH}}"
      timeout --signal=TERM --kill-after=60s 43200s
      xvfb-run -a -s "-screen 0 1024x768x24 -ac +extension GLX +render -noreset"
      "${PYTHON}" "${RUNTIME}/dreamerv3/main.py"
      --logdir "${TRAIN_DIR}" --configs minecraft size50m
      --task minecraft_diamond --script train --seed "${SEED}"
      --run.envs 1 --run.steps "${STEPS}" --run.train_ratio "${TRAIN_RATIO}"
      --run.log_every "${LOG_EVERY}" --run.report_every "${REPORT_EVERY}"
      --run.save_every "${SAVE_EVERY}" --run.save_at_end True)
    ;;
esac

printf 'Run directory: %s\nCommand:\n' "${RUN_DIR}"
printf ' %q' "${COMMAND[@]}"
printf '\n'
if ${DRY_RUN}; then
  exit 0
fi

preflight
mkdir "${RUN_DIR}"
EXECUTION_DIR=${RUN_DIR}
if [[ "${TASK}" == "minecraft" ]]; then
  EXECUTION_DIR=${RUN_DIR}/work
  mkdir "${EXECUTION_DIR}"
fi
{
  printf 'task=%s\nstage=%s\ncreated_at=%s\n' "${TASK}" "${STAGE}" "${TIMESTAMP}"
  printf 'control_commit=%s\nruntime_commit=%s\n' \
    "$(git -C "${CONTROL}" rev-parse HEAD)" "$(git -C "${RUNTIME}" rev-parse HEAD)"
  printf 'python=%s\nruntime_steps=%s\nstep_unit=%s\nenvironment_frames=%s\nseed=%s\ntrain_ratio=%s\n' \
    "${PYTHON}" "${STEPS}" "${STEP_UNIT}" "$(( STEPS * ENV_STEP_MULTIPLIER ))" \
    "${SEED}" "${TRAIN_RATIO}"
  printf 'command='
  printf ' %q' "${COMMAND[@]}"
  printf '\n'
} > "${RUN_DIR}/manifest.txt"

set +e
(cd "${EXECUTION_DIR}" && "${COMMAND[@]}") > "${STDOUT_LOG}" 2>&1
status=$?
set -e
if (( status != 0 )); then
  printf 'exit_code=%d\nfailed_at=%s\n' "${status}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    > "${RUN_DIR}/FAILED"
  echo "Run failed; see ${STDOUT_LOG}" >&2
  exit "${status}"
fi

if [[ "${STAGE}" == "l0" ]]; then
  printf 'completed_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${RUN_DIR}/COMPLETED"
  echo "Environment check completed: ${RUN_DIR}/l0"
  exit 0
fi

VERIFY_ARGS=(--task "${TASK}" --run-dir "${RUN_DIR}" --expected-step "${STEPS}"
  --stdout-log "${STDOUT_LOG}" --output "${RUN_DIR}/integrity.json")
if [[ "${TASK}" == "minecraft" ]]; then
  VERIFY_ARGS+=(--driver-step-quantum 10)
fi
"${PYTHON}" "${VERIFY}" "${VERIFY_ARGS[@]}"
printf 'completed_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${RUN_DIR}/COMPLETED"
echo "Run and integrity checks completed: ${RUN_DIR}"
