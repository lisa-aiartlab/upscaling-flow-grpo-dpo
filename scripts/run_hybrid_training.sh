#!/usr/bin/env bash
# Hybrid alignment from the research plan: DPO LoRA → online Flow-GRPO fine-tune.
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

DPO_CHECKPOINT="${DPO_CHECKPOINT:-dpo_output/final}"
if [[ -d "${FLOW_GRPO_PROJECT_DIR}/${DPO_CHECKPOINT}" ]]; then
  DPO_CHECKPOINT="${FLOW_GRPO_PROJECT_DIR}/${DPO_CHECKPOINT}"
fi
if [[ ! -e "${DPO_CHECKPOINT}" ]]; then
  echo "DPO checkpoint not found: ${DPO_CHECKPOINT}" >&2
  echo "Train DPO first: bash scripts/run_dpo.sh" >&2
  exit 1
fi

FLOW_GRPO_OUTPUT_DIR="${FLOW_GRPO_OUTPUT_DIR:-hybrid_flow_grpo_output}" \
FLOW_GRPO_INIT_FROM_LORA="${FLOW_GRPO_INIT_FROM_LORA:-${DPO_CHECKPOINT}}" \
  bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_training.sh" "$@"
