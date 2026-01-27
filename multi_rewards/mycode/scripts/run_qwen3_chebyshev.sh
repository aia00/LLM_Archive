#!/usr/bin/env bash
set -euo pipefail

METHOD=chebyshev \
CHEBYSHEV_REFERENCE=0,0,0 \
bash "$(dirname "$0")/run_train_multiobj.sh"
