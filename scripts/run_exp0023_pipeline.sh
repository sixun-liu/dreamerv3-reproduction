#!/usr/bin/env bash
set -euo pipefail

readonly CONTROL=/root/autodl-tmp/Code/DreamerV3/dreamerv3-reproduction

"${CONTROL}/scripts/run_exp0023_train.sh"
"${CONTROL}/scripts/run_exp0023_parallel_eval.sh"
"${CONTROL}/scripts/postprocess_exp0023.sh"
