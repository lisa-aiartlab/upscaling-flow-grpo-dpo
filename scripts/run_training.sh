#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

if [[ ! -x "${FLOW_GRPO_PYTHON}" ]]; then
  echo "Virtual environment not found. Run: bash scripts/setup_vm.sh" >&2
  exit 1
fi

cd "${FLOW_GRPO_PROJECT_DIR}"

FLOW_GRPO_MANIFEST="${FLOW_GRPO_MANIFEST:-flow_grpo_dataset/upscaling_dataset/manifest.json}"
FLOW_GRPO_OUTPUT_DIR="${FLOW_GRPO_OUTPUT_DIR:-flow_grpo_output}"
FLOW_GRPO_EPOCHS="${FLOW_GRPO_EPOCHS:-1}"
FLOW_GRPO_GROUP_SIZE="${FLOW_GRPO_GROUP_SIZE:-4}"
FLOW_GRPO_INFERENCE_STEPS="${FLOW_GRPO_INFERENCE_STEPS:-4}"
FLOW_GRPO_RESOLUTION="${FLOW_GRPO_RESOLUTION:-120}"
FLOW_GRPO_MIXED_PRECISION="${FLOW_GRPO_MIXED_PRECISION:-bf16}"
FLOW_GRPO_REWARD_DEVICE="${FLOW_GRPO_REWARD_DEVICE:-cpu}"

training_args=(
  --manifest "${FLOW_GRPO_MANIFEST}"
  --output-dir "${FLOW_GRPO_OUTPUT_DIR}"
  --epochs "${FLOW_GRPO_EPOCHS}"
  --grpo-epochs 1
  --group-size "${FLOW_GRPO_GROUP_SIZE}"
  --inference-steps "${FLOW_GRPO_INFERENCE_STEPS}"
  --resolution "${FLOW_GRPO_RESOLUTION}"
  --save-every 25
  --reward-device "${FLOW_GRPO_REWARD_DEVICE}"
  --mixed-precision "${FLOW_GRPO_MIXED_PRECISION}"
)

if [[ -n "${FLOW_GRPO_LR_PIPELINE:-}" ]]; then
  training_args+=(--lr-pipeline "${FLOW_GRPO_LR_PIPELINE}")
fi
if [[ -n "${FLOW_GRPO_MAX_SAMPLES:-}" ]]; then
  training_args+=(--max-samples "${FLOW_GRPO_MAX_SAMPLES}")
fi

exec "${FLOW_GRPO_PYTHON}" scripts/training_scripts/flow_grpo.py \
  "${training_args[@]}" "$@"
