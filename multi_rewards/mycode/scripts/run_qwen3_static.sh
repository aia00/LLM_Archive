#!/usr/bin/env bash
set -euo pipefail

METHOD=static WEIGHT_PRESET=balanced \
bash "$(dirname "$0")/run_verl_multiobj.sh"
