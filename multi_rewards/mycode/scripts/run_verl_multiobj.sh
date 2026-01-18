#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
VERL_ROOT="${PROJECT_ROOT}/verl"
export PYTHONPATH="${PROJECT_ROOT}:${VERL_ROOT}:${PYTHONPATH:-}"

CONFIG_PATH=${CONFIG_PATH:-"${PROJECT_ROOT}/mycode/paths.env"}
if [ -f "${CONFIG_PATH}" ]; then
  # shellcheck disable=SC1090
  source "${CONFIG_PATH}"
fi

if [ -n "${GPU_IDS:-}" ]; then
  export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
fi

if [ -n "${RAY_TMPDIR:-}" ]; then
  export RAY_TMPDIR
fi


CONDA_ENV_PATH=${CONDA_ENV_PATH:-""}
CONDA_ENV_NAME=${CONDA_ENV_NAME:-""}
CONDA_SH=${CONDA_SH:-"${HOME}/.conda/etc/profile.d/conda.sh"}
PYTHON_BIN="python"
if [ -n "${CONDA_ENV_NAME}" ]; then
  # Prefer the exact activation pattern that works in this environment.
  set +u
  source activate "${CONDA_ENV_NAME}" || conda activate "${CONDA_ENV_NAME}"
  set -u
elif [ -n "${CONDA_ENV_PATH}" ]; then
  if [ -f "${CONDA_SH}" ]; then
    # Activate env so VERL deps resolve consistently.
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
OUTPUT_ROOT=${OUTPUT_ROOT:-"${PROJECT_ROOT}/outputs"}

DATASET_PRESET=${DATASET_PRESET:-"math500"}
TRAIN_PARQUET=${TRAIN_PARQUET:-""}
VAL_PARQUET=${VAL_PARQUET:-""}
if [ -z "${TRAIN_PARQUET}" ] || [ -z "${VAL_PARQUET}" ]; then
  if [ "${DATASET_PRESET}" = "math_full" ]; then
    TRAIN_PARQUET=${TRAIN_PARQUET:-"${DATA_DIR}/math_full_train.parquet"}
    VAL_PARQUET=${VAL_PARQUET:-"${DATA_DIR}/math_full_val.parquet"}
  else
    TRAIN_PARQUET=${TRAIN_PARQUET:-"${DATA_DIR}/math_train.parquet"}
    VAL_PARQUET=${VAL_PARQUET:-"${DATA_DIR}/math_val.parquet"}
  fi
fi
MODEL_NAME=${MODEL_NAME:-${MODEL_NAME_DEFAULT:-"Qwen/Qwen2.5-0.5B-Instruct"}}
METHOD=${METHOD:-"static"}
WEIGHT_PRESET=${WEIGHT_PRESET:-"balanced"}
REWARD_NAMES=${REWARD_NAMES:-"reward_acc,reward_conc,reward_clar,reward_answer_len,reward_answer_found,reward_boxed"}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-16}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-${TRAIN_BATCH_SIZE}}
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}
PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}
ATTN_IMPL=${ATTN_IMPL:-"sdpa"}
N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-4}
TENSOR_MODEL_PARALLEL_SIZE=${TENSOR_MODEL_PARALLEL_SIZE:-${N_GPUS_PER_NODE}}
ROLLOUT_NAME=${ROLLOUT_NAME:-"vllm"}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.30}
USE_REMOVE_PADDING=${USE_REMOVE_PADDING:-false}
OFFLOAD_PARAM=${OFFLOAD_PARAM:-true}
OFFLOAD_OPTIMIZER=${OFFLOAD_OPTIMIZER:-true}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-512}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-1024}
ROLLOUT_N=${ROLLOUT_N:-8}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-1024}
PARETO_REWARD_NAMES=${PARETO_REWARD_NAMES:-"reward_acc,reward_conc,reward_clar"}
PARETO_HV_SAMPLES=${PARETO_HV_SAMPLES:-4096}
PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-""}
RUN_TAG=${RUN_TAG:-"$(date +%Y%m%d_%H%M%S)"}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-"${METHOD}_${WEIGHT_PRESET}_${RUN_TAG}"}
ROLLOUT_DATA_DIR=${ROLLOUT_DATA_DIR:-"${OUTPUT_ROOT}/rollouts/${EXPERIMENT_NAME}"}
VALIDATION_DATA_DIR=${VALIDATION_DATA_DIR:-"${OUTPUT_ROOT}/val_generations/${EXPERIMENT_NAME}"}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-3}
TEST_FREQ=${TEST_FREQ:-5}

if [ -n "${PYTORCH_CUDA_ALLOC_CONF}" ]; then
  export PYTORCH_CUDA_ALLOC_CONF
fi

TOTAL_LEN=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))
if [ "${MAX_MODEL_LEN}" -lt "${TOTAL_LEN}" ]; then
  MAX_MODEL_LEN="${TOTAL_LEN}"
fi

REWARD_NAMES_HYDRA="${REWARD_NAMES}"
if [[ "${REWARD_NAMES}" != \[* ]]; then
  IFS=',' read -r -a _reward_arr <<< "${REWARD_NAMES}"
  REWARD_NAMES_HYDRA="["
  for _name in "${_reward_arr[@]}"; do
    _trimmed=$(echo "${_name}" | xargs)
    if [ -n "${_trimmed}" ]; then
      REWARD_NAMES_HYDRA+="'${_trimmed}',"
    fi
  done
  REWARD_NAMES_HYDRA="${REWARD_NAMES_HYDRA%,}]"
fi

PARETO_REWARD_NAMES_HYDRA="${PARETO_REWARD_NAMES}"
if [ -n "${PARETO_REWARD_NAMES}" ] && [[ "${PARETO_REWARD_NAMES}" != \[* ]]; then
  IFS=',' read -r -a _pareto_arr <<< "${PARETO_REWARD_NAMES}"
  PARETO_REWARD_NAMES_HYDRA="["
  for _name in "${_pareto_arr[@]}"; do
    _trimmed=$(echo "${_name}" | xargs)
    if [ -n "${_trimmed}" ]; then
      PARETO_REWARD_NAMES_HYDRA+="'${_trimmed}',"
    fi
  done
  PARETO_REWARD_NAMES_HYDRA="${PARETO_REWARD_NAMES_HYDRA%,}]"
fi

${PYTHON_BIN} -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files="['${TRAIN_PARQUET}']" \
  data.val_files="['${VAL_PARQUET}']" \
  data.train_batch_size=${TRAIN_BATCH_SIZE} \
  data.max_prompt_length=${MAX_PROMPT_LENGTH} \
  data.max_response_length=${MAX_RESPONSE_LENGTH} \
  reward_manager.name=batch \
  custom_reward_function.path="${PROJECT_ROOT}/mycode/verl_reward.py" \
  custom_reward_function.name=compute_score \
  +custom_reward_function.reward_kwargs.multi_objective_mode=${METHOD} \
  +custom_reward_function.reward_kwargs.weight_preset=${WEIGHT_PRESET} \
  +custom_reward_function.reward_kwargs.reward_names="${REWARD_NAMES_HYDRA}" \
  $(if [ -n "${PARETO_REWARD_NAMES_HYDRA}" ]; then echo "+custom_reward_function.reward_kwargs.pareto_reward_names=${PARETO_REWARD_NAMES_HYDRA}"; fi) \
  +custom_reward_function.reward_kwargs.pareto_hv_samples=${PARETO_HV_SAMPLES} \
  +custom_reward_function.reward_kwargs.dynamic_eta=0.5 \
  +custom_reward_function.reward_kwargs.dynamic_mu=1.0 \
  +custom_reward_function.reward_kwargs.pareto_max_size=128 \
  actor_rollout_ref.model.path=${MODEL_NAME} \
  +actor_rollout_ref.model.override_config.attn_implementation=${ATTN_IMPL} \
  actor_rollout_ref.rollout.name=${ROLLOUT_NAME} \
  actor_rollout_ref.rollout.n=${ROLLOUT_N} \
  actor_rollout_ref.rollout.tensor_model_parallel_size=${TENSOR_MODEL_PARALLEL_SIZE} \
  actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEMORY_UTILIZATION} \
  actor_rollout_ref.rollout.prompt_length=${MAX_PROMPT_LENGTH} \
  actor_rollout_ref.rollout.response_length=${MAX_RESPONSE_LENGTH} \
  actor_rollout_ref.rollout.max_model_len=${MAX_MODEL_LEN} \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU} \
  actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE} \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE_PER_GPU} \
  actor_rollout_ref.actor.fsdp_config.param_offload=${OFFLOAD_PARAM} \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=${OFFLOAD_OPTIMIZER} \
  actor_rollout_ref.actor.use_remove_padding=${USE_REMOVE_PADDING} \
  actor_rollout_ref.model.use_remove_padding=${USE_REMOVE_PADDING} \
  trainer.project_name=verl_multiobj_math \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  trainer.default_local_dir="${OUTPUT_ROOT}/checkpoints/${EXPERIMENT_NAME}" \
  trainer.rollout_data_dir="${ROLLOUT_DATA_DIR}" \
  trainer.validation_data_dir="${VALIDATION_DATA_DIR}" \
  trainer.n_gpus_per_node=${N_GPUS_PER_NODE} \
  trainer.nnodes=1 \
  trainer.total_epochs=${TOTAL_EPOCHS} \
  trainer.test_freq=${TEST_FREQ}
