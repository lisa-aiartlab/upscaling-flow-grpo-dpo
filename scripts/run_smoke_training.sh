#!/usr/bin/env bash
set -euo pipefail

FLOW_GRPO_MAX_SAMPLES=1 \
FLOW_GRPO_GROUP_SIZE=2 \
FLOW_GRPO_INFERENCE_STEPS=4 \
FLOW_GRPO_LR_PIPELINE="${FLOW_GRPO_LR_PIPELINE:-LR_01_resize}" \
FLOW_GRPO_OUTPUT_DIR="${FLOW_GRPO_OUTPUT_DIR:-flow_grpo_smoke_output}" \
  bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_training.sh" \
    --save-every 1 \
    --clip-model-id none \
    "$@"
