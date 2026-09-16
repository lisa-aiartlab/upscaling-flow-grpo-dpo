#!/usr/bin/env bash

# Shared Linux environment for training and inference launchers.
FLOW_GRPO_PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FLOW_GRPO_CACHE_DIR="${FLOW_GRPO_CACHE_DIR:-${FLOW_GRPO_PROJECT_DIR}/.cache}"
FLOW_GRPO_PYTHON="${FLOW_GRPO_PYTHON:-${FLOW_GRPO_PROJECT_DIR}/.venv/bin/python}"

export HF_HOME="${HF_HOME:-${FLOW_GRPO_CACHE_DIR}/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export TORCH_HOME="${TORCH_HOME:-${FLOW_GRPO_CACHE_DIR}/torch}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-${FLOW_GRPO_CACHE_DIR}/cuda}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${FLOW_GRPO_CACHE_DIR}/pip}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${FLOW_GRPO_CACHE_DIR}}"
export TMPDIR="${TMPDIR:-${FLOW_GRPO_CACHE_DIR}/tmp}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

mkdir -p \
  "${HF_HUB_CACHE}" \
  "${TORCH_HOME}" \
  "${CUDA_CACHE_PATH}" \
  "${PIP_CACHE_DIR}" \
  "${TMPDIR}"
