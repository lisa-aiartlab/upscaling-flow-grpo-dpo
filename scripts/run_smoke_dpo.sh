#!/usr/bin/env bash
set -euo pipefail

DPO_MAX_SAMPLES=1 \
DPO_MANIFEST="${DPO_MANIFEST:-flow_grpo_dataset/manifest.json}" \
DPO_OUTPUT_DIR="${DPO_OUTPUT_DIR:-dpo_smoke_output}" \
DPO_BETA="${DPO_BETA:-500}" \
  bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_dpo.sh" \
    --save-every 1 \
    "$@"
