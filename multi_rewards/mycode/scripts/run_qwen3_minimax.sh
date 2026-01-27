#!/usr/bin/env bash
set -euo pipefail

METHOD=minimax \
MINIMAX_ETA=0.5 \
bash "$(dirname "$0")/run_train_multiobj.sh"
