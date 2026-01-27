#!/usr/bin/env bash
set -euo pipefail

METHOD=pcgrad \
PCGRAD_SHUFFLE=1 \
bash "$(dirname "$0")/run_train_multiobj.sh"
