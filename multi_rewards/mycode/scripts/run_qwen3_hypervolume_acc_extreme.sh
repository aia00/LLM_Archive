#!/usr/bin/env bash
set -euo pipefail

METHOD=hypervolume \
WEIGHT_PRESET=balanced \
REWARD_NAMES="reward_acc,reward_conc,reward_clar,reward_answer_len,reward_answer_found,reward_boxed" \
WEIGHTS="0.95,0.01,0.01,0.01,0.01,0.01" \
bash "$(dirname "$0")/run_verl_multiobj.sh"
