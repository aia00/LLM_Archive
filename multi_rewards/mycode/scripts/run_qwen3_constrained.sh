#!/usr/bin/env bash
set -euo pipefail

METHOD=constrained \
PRIMARY_REWARD=accuracy \
CONSTRAINT_REWARDS=conciseness,clarity \
CONSTRAINT_THRESHOLDS=0.2,0.2 \
CONSTRAINT_DIRECTIONS=max,max \
bash "$(dirname "$0")/run_train_multiobj.sh"
