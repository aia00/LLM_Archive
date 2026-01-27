#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

CONFIG_PATH=${CONFIG_PATH:-"${PROJECT_ROOT}/mycode/paths.env"}
if [ -f "${CONFIG_PATH}" ]; then
  # shellcheck disable=SC1090
  source "${CONFIG_PATH}"
fi

if [ -n "${WANDB_API_KEY:-}" ]; then
  export WANDB_API_KEY
fi

MODEL_NAME=${MODEL_NAME:-"Qwen/Qwen2.5-0.5B-Instruct"}
TOKENIZER_NAME=${TOKENIZER_NAME:-""}
DATASET_NAME=${DATASET_NAME:-"HuggingFaceH4/MATH-500"}
TRAIN_SPLIT=${TRAIN_SPLIT:-"train"}
EVAL_SPLIT=${EVAL_SPLIT:-""}
OUTPUT_DIR=${OUTPUT_DIR:-"${PROJECT_ROOT}/outputs/train_multiobj"}

METHOD=${METHOD:-"static"}
WEIGHT_PRESET=${WEIGHT_PRESET:-"balanced"}
WEIGHT_ACCURACY=${WEIGHT_ACCURACY:-""}
WEIGHT_CONCISENESS=${WEIGHT_CONCISENESS:-""}
WEIGHT_CLARITY=${WEIGHT_CLARITY:-""}

PRIMARY_REWARD=${PRIMARY_REWARD:-"accuracy"}
CONSTRAINT_REWARDS=${CONSTRAINT_REWARDS:-""}
CONSTRAINT_THRESHOLDS=${CONSTRAINT_THRESHOLDS:-""}
CONSTRAINT_DIRECTIONS=${CONSTRAINT_DIRECTIONS:-""}
CONSTRAINT_LAMBDA_LR=${CONSTRAINT_LAMBDA_LR:-""}
CONSTRAINT_LAMBDA_MAX=${CONSTRAINT_LAMBDA_MAX:-""}
CONSTRAINT_DISABLE_EMA=${CONSTRAINT_DISABLE_EMA:-""}
CONSTRAINT_EMA_ALPHA=${CONSTRAINT_EMA_ALPHA:-""}

MINIMAX_ETA=${MINIMAX_ETA:-""}
MINIMAX_MIN_WEIGHT=${MINIMAX_MIN_WEIGHT:-""}

CHEBYSHEV_REFERENCE=${CHEBYSHEV_REFERENCE:-""}
NASH_BASELINE=${NASH_BASELINE:-""}
NASH_EPSILON=${NASH_EPSILON:-""}
CVAR_ALPHA=${CVAR_ALPHA:-""}

CONTEXT_SHORT_LENGTH=${CONTEXT_SHORT_LENGTH:-""}
CONTEXT_LONG_LENGTH=${CONTEXT_LONG_LENGTH:-""}
CONTEXT_SHORT_WEIGHTS=${CONTEXT_SHORT_WEIGHTS:-""}
CONTEXT_LONG_WEIGHTS=${CONTEXT_LONG_WEIGHTS:-""}
CONTEXT_CLARITY_WEIGHTS=${CONTEXT_CLARITY_WEIGHTS:-""}
CONTEXT_CONCISENESS_WEIGHTS=${CONTEXT_CONCISENESS_WEIGHTS:-""}

UNCERTAINTY_EMA_ALPHA=${UNCERTAINTY_EMA_ALPHA:-""}
UNCERTAINTY_EPS=${UNCERTAINTY_EPS:-""}

PCGRAD_SHUFFLE=${PCGRAD_SHUFFLE:-""}
CAGRAD_ALPHA=${CAGRAD_ALPHA:-""}
MGDA_ITERS=${MGDA_ITERS:-""}

NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS:-"1"}
MAX_STEPS=${MAX_STEPS:-"-1"}
PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-"1"}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-"16"}
LEARNING_RATE=${LEARNING_RATE:-"1e-6"}
NUM_GENERATIONS=${NUM_GENERATIONS:-"8"}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-"2048"}
MAX_COMPLETION_LENGTH=${MAX_COMPLETION_LENGTH:-"4096"}
BF16=${BF16:-""}
REPORT_TO=${REPORT_TO:-""}

ARGS=(
  "--model_name_or_path" "${MODEL_NAME}"
  "--dataset_name" "${DATASET_NAME}"
  "--train_split" "${TRAIN_SPLIT}"
  "--method" "${METHOD}"
  "--weight_preset" "${WEIGHT_PRESET}"
  "--output_dir" "${OUTPUT_DIR}"
  "--num_train_epochs" "${NUM_TRAIN_EPOCHS}"
  "--max_steps" "${MAX_STEPS}"
  "--per_device_train_batch_size" "${PER_DEVICE_TRAIN_BATCH_SIZE}"
  "--gradient_accumulation_steps" "${GRADIENT_ACCUMULATION_STEPS}"
  "--learning_rate" "${LEARNING_RATE}"
  "--num_generations" "${NUM_GENERATIONS}"
  "--max_prompt_length" "${MAX_PROMPT_LENGTH}"
  "--max_completion_length" "${MAX_COMPLETION_LENGTH}"
)

if [ -n "${TOKENIZER_NAME}" ]; then
  ARGS+=("--tokenizer_name" "${TOKENIZER_NAME}")
fi
if [ -n "${EVAL_SPLIT}" ]; then
  ARGS+=("--eval_split" "${EVAL_SPLIT}")
fi
if [ -n "${WEIGHT_ACCURACY}" ]; then
  ARGS+=("--weight_accuracy" "${WEIGHT_ACCURACY}")
fi
if [ -n "${WEIGHT_CONCISENESS}" ]; then
  ARGS+=("--weight_conciseness" "${WEIGHT_CONCISENESS}")
fi
if [ -n "${WEIGHT_CLARITY}" ]; then
  ARGS+=("--weight_clarity" "${WEIGHT_CLARITY}")
fi

if [ -n "${PRIMARY_REWARD}" ]; then
  ARGS+=("--primary_reward" "${PRIMARY_REWARD}")
fi
if [ -n "${CONSTRAINT_REWARDS}" ]; then
  ARGS+=("--constraint_rewards" "${CONSTRAINT_REWARDS}")
fi
if [ -n "${CONSTRAINT_THRESHOLDS}" ]; then
  ARGS+=("--constraint_thresholds" "${CONSTRAINT_THRESHOLDS}")
fi
if [ -n "${CONSTRAINT_DIRECTIONS}" ]; then
  ARGS+=("--constraint_directions" "${CONSTRAINT_DIRECTIONS}")
fi
if [ -n "${CONSTRAINT_LAMBDA_LR}" ]; then
  ARGS+=("--constraint_lambda_lr" "${CONSTRAINT_LAMBDA_LR}")
fi
if [ -n "${CONSTRAINT_LAMBDA_MAX}" ]; then
  ARGS+=("--constraint_lambda_max" "${CONSTRAINT_LAMBDA_MAX}")
fi
if [ -n "${CONSTRAINT_DISABLE_EMA}" ]; then
  ARGS+=("--constraint_disable_ema")
fi
if [ -n "${CONSTRAINT_EMA_ALPHA}" ]; then
  ARGS+=("--constraint_ema_alpha" "${CONSTRAINT_EMA_ALPHA}")
fi

if [ -n "${MINIMAX_ETA}" ]; then
  ARGS+=("--minimax_eta" "${MINIMAX_ETA}")
fi
if [ -n "${MINIMAX_MIN_WEIGHT}" ]; then
  ARGS+=("--minimax_min_weight" "${MINIMAX_MIN_WEIGHT}")
fi

if [ -n "${CHEBYSHEV_REFERENCE}" ]; then
  ARGS+=("--chebyshev_reference" "${CHEBYSHEV_REFERENCE}")
fi
if [ -n "${NASH_BASELINE}" ]; then
  ARGS+=("--nash_baseline" "${NASH_BASELINE}")
fi
if [ -n "${NASH_EPSILON}" ]; then
  ARGS+=("--nash_epsilon" "${NASH_EPSILON}")
fi
if [ -n "${CVAR_ALPHA}" ]; then
  ARGS+=("--cvar_alpha" "${CVAR_ALPHA}")
fi

if [ -n "${CONTEXT_SHORT_LENGTH}" ]; then
  ARGS+=("--context_short_length" "${CONTEXT_SHORT_LENGTH}")
fi
if [ -n "${CONTEXT_LONG_LENGTH}" ]; then
  ARGS+=("--context_long_length" "${CONTEXT_LONG_LENGTH}")
fi
if [ -n "${CONTEXT_SHORT_WEIGHTS}" ]; then
  ARGS+=("--context_short_weights" "${CONTEXT_SHORT_WEIGHTS}")
fi
if [ -n "${CONTEXT_LONG_WEIGHTS}" ]; then
  ARGS+=("--context_long_weights" "${CONTEXT_LONG_WEIGHTS}")
fi
if [ -n "${CONTEXT_CLARITY_WEIGHTS}" ]; then
  ARGS+=("--context_clarity_weights" "${CONTEXT_CLARITY_WEIGHTS}")
fi
if [ -n "${CONTEXT_CONCISENESS_WEIGHTS}" ]; then
  ARGS+=("--context_conciseness_weights" "${CONTEXT_CONCISENESS_WEIGHTS}")
fi

if [ -n "${UNCERTAINTY_EMA_ALPHA}" ]; then
  ARGS+=("--uncertainty_ema_alpha" "${UNCERTAINTY_EMA_ALPHA}")
fi
if [ -n "${UNCERTAINTY_EPS}" ]; then
  ARGS+=("--uncertainty_eps" "${UNCERTAINTY_EPS}")
fi

if [ -n "${PCGRAD_SHUFFLE}" ]; then
  ARGS+=("--pcgrad_shuffle")
fi
if [ -n "${CAGRAD_ALPHA}" ]; then
  ARGS+=("--cagrad_alpha" "${CAGRAD_ALPHA}")
fi
if [ -n "${MGDA_ITERS}" ]; then
  ARGS+=("--mgda_iters" "${MGDA_ITERS}")
fi

if [ -n "${BF16}" ]; then
  ARGS+=("--bf16")
fi
if [ -n "${REPORT_TO}" ]; then
  ARGS+=("--report_to" "${REPORT_TO}")
fi

python "${PROJECT_ROOT}/mycode/train_multi_objective.py" "${ARGS[@]}"
