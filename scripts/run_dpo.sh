#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

if [[ ! -x "${FLOW_GRPO_PYTHON}" ]]; then
  echo "Virtual environment not found. Run: bash scripts/setup_vm.sh" >&2
  exit 1
fi

cd "${FLOW_GRPO_PROJECT_DIR}"

DPO_MANIFEST="${DPO_MANIFEST:-flow_grpo_dataset/manifest.json}"
DPO_OUTPUT_DIR="${DPO_OUTPUT_DIR:-dpo_output}"
DPO_EPOCHS="${DPO_EPOCHS:-1}"
DPO_RESOLUTION="${DPO_RESOLUTION:-128}"
DPO_MIXED_PRECISION="${DPO_MIXED_PRECISION:-${FLOW_GRPO_MIXED_PRECISION:-fp16}}"
DPO_BETA="${DPO_BETA:-500}"
DPO_LORA_RANK="${DPO_LORA_RANK:-4}"
DPO_LEARNING_RATE="${DPO_LEARNING_RATE:-1.0e-5}"
DPO_SYNTHETIC_REJECTED="${DPO_SYNTHETIC_REJECTED:-lanczos}"

training_args=(
  --manifest "${DPO_MANIFEST}"
  --output-dir "${DPO_OUTPUT_DIR}"
  --epochs "${DPO_EPOCHS}"
  --resolution "${DPO_RESOLUTION}"
  --beta "${DPO_BETA}"
  --lora-rank "${DPO_LORA_RANK}"
  --learning-rate "${DPO_LEARNING_RATE}"
  --synthetic-rejected "${DPO_SYNTHETIC_REJECTED}"
  --save-every 25
  --mixed-precision "${DPO_MIXED_PRECISION}"
)

if [[ -n "${DPO_LR_PIPELINE:-}" ]]; then
  training_args+=(--lr-pipeline "${DPO_LR_PIPELINE}")
fi
if [[ -n "${DPO_MAX_SAMPLES:-}" ]]; then
  training_args+=(--max-samples "${DPO_MAX_SAMPLES}")
fi
if [[ -n "${DPO_RESUME_FROM_CHECKPOINT:-}" ]]; then
  training_args+=(--resume-from-checkpoint "${DPO_RESUME_FROM_CHECKPOINT}")
fi
if [[ -n "${DPO_INIT_FROM_LORA:-}" ]]; then
  training_args+=(--init-from-lora "${DPO_INIT_FROM_LORA}")
fi

exec "${FLOW_GRPO_PYTHON}" scripts/training_scripts/dpo.py \
  "${training_args[@]}" "$@"
