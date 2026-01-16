#!/usr/bin/env bash
set -euo pipefail

METHOD=hypervolume WEIGHT_PRESET=balanced \
bash "$(dirname "$0")/run_verl_multiobj.sh"
