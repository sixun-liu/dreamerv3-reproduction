#!/usr/bin/env bash
set -euo pipefail

readonly CONTROL=/root/autodl-tmp/Code/DreamerV3/dreamerv3-reproduction
readonly EXPERIMENT=EXP-0020
readonly TRAIN=/root/autodl-tmp/Runs/EXP-0020__minecraft-diamond__s000__100k-to-200k-env__20260812T100000Z
readonly EVAL=/root/autodl-tmp/Runs/EXP-0020__minecraft-diamond__eval-s10000-3eps__20260812T100000Z
readonly REVIEW=/root/autodl-tmp/Artifacts/dreamerv3/review/EXP-0020-minecraft-exact-200k-increment

export EXPERIMENT_ID=${EXPERIMENT}
export DV3_RUNTIME=/root/autodl-tmp/Code/DreamerV3/dreamerv3-exp0018-exact-stop
export DV3_RUNTIME_COMMIT=6723fc1620636cc8a52e0ddc0b7a9cbe946625b0
export TRAIN_RUN_ROOT=${TRAIN}
export EVAL_RUN_ROOT=${EVAL}
export TRAIN_CONFIG=${CONTROL}/docs/reproduction/configs/exp0020_minecraft_s000_200k_env.yaml
export EXPERIMENT_MATRIX=${CONTROL}/docs/reproduction/configs/exp0020_minecraft_200k_matrix.yaml
export REVIEW_OUTPUT=${REVIEW}

"${CONTROL}/scripts/run_exp0017_train.sh"
"${CONTROL}/scripts/run_exp0017_eval.sh"
"${CONTROL}/scripts/postprocess_exp0017.sh"
