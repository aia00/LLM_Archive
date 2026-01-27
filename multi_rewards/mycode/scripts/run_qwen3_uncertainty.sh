#!/usr/bin/env bash
set -euo pipefail

METHOD=uncertainty \
UNCERTAINTY_EMA_ALPHA=0.05 \
UNCERTAINTY_EPS=1e-6 \
bash "$(dirname "$0")/run_train_multiobj.sh"
