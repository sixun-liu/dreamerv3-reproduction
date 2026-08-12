#!/usr/bin/env bash
set -euo pipefail

readonly CONTROL=/root/autodl-tmp/Code/DreamerV3/dreamerv3-reproduction
readonly EXPERIMENT=EXP-0023
readonly ROOT=/root/autodl-tmp/Runs/EXP-0023__minecraft-diamond__s000__200k-to-500k-env__20260812T123000Z

export EXPERIMENT_ID=${EXPERIMENT}
export DV3_RUNTIME=/root/autodl-tmp/Code/DreamerV3/dreamerv3-exp0018-exact-stop
export DV3_RUNTIME_COMMIT=6723fc1620636cc8a52e0ddc0b7a9cbe946625b0
export SOURCE_TRAIN=/root/autodl-tmp/Runs/EXP-0020__minecraft-diamond__s000__100k-to-200k-env__20260812T100000Z/train
export SOURCE_CONFIG=${CONTROL}/docs/reproduction/configs/exp0020_minecraft_s000_200k_env.yaml
export TRAIN_RUN_ROOT=${ROOT}
export TRAIN_CONFIG=${CONTROL}/docs/reproduction/configs/exp0023_minecraft_s000_500k_env.yaml
export EXPERIMENT_MATRIX=${CONTROL}/docs/reproduction/configs/exp0023_minecraft_500k_matrix.yaml
export SOURCE_STEP_OVERRIDE=200000
export FINAL_STEP_OVERRIDE=500000
export ENVIRONMENT_COUNT_OVERRIDE=4
export SOURCE_CHECKPOINT_TREE_SHA256_OVERRIDE=dea72ee54c7c36be275026d0140902c706270af98b278fc61e54c190e5538865
export SOURCE_REPLAY_TREE_SHA256_OVERRIDE=3b78b03266d7564ed33c4bd80dd9eab760ce11f5e311149db167d407900198ad
export CONFIG_GENERATOR_OVERRIDE=${CONTROL}/scripts/generate_exp0023_config.py
export REQUIRED_CONFIG_CHANGES_OVERRIDE=logdir,run.steps
export TRAIN_TIMEOUT_SECONDS_OVERRIDE=14400

exec "${CONTROL}/scripts/run_exp0017_train.sh"
