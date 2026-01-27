#!/usr/bin/env bash
set -euo pipefail

METHOD=contextual \
CONTEXT_SHORT_WEIGHTS=0.4,0.4,0.2 \
CONTEXT_LONG_WEIGHTS=0.3,0.2,0.5 \
CONTEXT_CLARITY_WEIGHTS=0.3,0.2,0.5 \
CONTEXT_CONCISENESS_WEIGHTS=0.4,0.4,0.2 \
bash "$(dirname "$0")/run_train_multiobj.sh"
