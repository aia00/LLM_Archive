#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
CONFIG_PATH=${CONFIG_PATH:-"${PROJECT_ROOT}/mycode/paths.env"}

if [ -f "${CONFIG_PATH}" ]; then
  # shellcheck disable=SC1090
  source "${CONFIG_PATH}"
fi

CONDA_ENV_PATH=${CONDA_ENV_PATH:-""}
CONDA_ENV_NAME=${CONDA_ENV_NAME:-""}
if [ -n "${CONDA_SH:-}" ]; then
  CONDA_SH="${CONDA_SH}"
elif [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
  CONDA_SH="${HOME}/miniconda3/etc/profile.d/conda.sh"
else
  CONDA_SH="${HOME}/.conda/etc/profile.d/conda.sh"
fi
PYTHON_BIN="python"
if [ -n "${CONDA_ENV_NAME}" ]; then
  if [ -f "${CONDA_SH}" ]; then
    set +u
    source "${CONDA_SH}"
    conda activate "${CONDA_ENV_NAME}"
    set -u
  else
    echo "Conda activation script not found at ${CONDA_SH}" >&2
    exit 1
  fi
elif [ -n "${CONDA_ENV_PATH}" ]; then
  if [ -f "${CONDA_SH}" ]; then
    set +u
    source "${CONDA_SH}"
    conda activate "${CONDA_ENV_PATH}"
    set -u
  else
    if [ -x "${CONDA_ENV_PATH}/bin/python" ]; then
      PYTHON_BIN="${CONDA_ENV_PATH}/bin/python"
      echo "Conda activation script not found at ${CONDA_SH}; using ${PYTHON_BIN}" >&2
    else
      echo "Conda activation script not found at ${CONDA_SH}" >&2
      exit 1
    fi
  fi
fi

DATA_DIR=${DATA_DIR:-"${PROJECT_ROOT}/data"}
DATASET_NAME=${DATASET_NAME:-"HuggingFaceH4/MATH-500"}
TRAIN_SPLIT=${TRAIN_SPLIT:-"test"}
EVAL_SPLIT=${EVAL_SPLIT:-"${TRAIN_SPLIT}"}
QUESTION_KEY=${QUESTION_KEY:-""}
ANSWER_KEY=${ANSWER_KEY:-""}
SYSTEM_PROMPT=${SYSTEM_PROMPT:-"You are a helpful math assistant."}
PROMPT_TEMPLATE=${PROMPT_TEMPLATE:-$'Problem:\n{question}\n\nAnswer:'}
MAX_TRAIN_SAMPLES=${MAX_TRAIN_SAMPLES:-""}
MAX_EVAL_SAMPLES=${MAX_EVAL_SAMPLES:-""}
TRAIN_OUT=${TRAIN_OUT:-"${DATA_DIR}/math_train.parquet"}
EVAL_OUT=${EVAL_OUT:-"${DATA_DIR}/math_val.parquet"}

mkdir -p "$(dirname "${TRAIN_OUT}")"
mkdir -p "$(dirname "${EVAL_OUT}")"

ARGS=(
  --dataset_name "${DATASET_NAME}"
  --train_split "${TRAIN_SPLIT}"
  --train_out "${TRAIN_OUT}"
  --system_prompt "${SYSTEM_PROMPT}"
  --prompt_template "${PROMPT_TEMPLATE}"
)

if [ -n "${EVAL_SPLIT}" ]; then
  ARGS+=(--eval_split "${EVAL_SPLIT}" --eval_out "${EVAL_OUT}")
fi
if [ -n "${QUESTION_KEY}" ]; then
  ARGS+=(--question_key "${QUESTION_KEY}")
fi
if [ -n "${ANSWER_KEY}" ]; then
  ARGS+=(--answer_key "${ANSWER_KEY}")
fi
if [ -n "${MAX_TRAIN_SAMPLES}" ]; then
  ARGS+=(--max_train_samples "${MAX_TRAIN_SAMPLES}")
fi
if [ -n "${MAX_EVAL_SAMPLES}" ]; then
  ARGS+=(--max_eval_samples "${MAX_EVAL_SAMPLES}")
fi

${PYTHON_BIN} "${PROJECT_ROOT}/mycode/prepare_verl_math.py" "${ARGS[@]}"
