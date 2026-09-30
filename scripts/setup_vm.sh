#!/usr/bin/env bash
set -euo pipefail

FLOW_GRPO_PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FLOW_GRPO_SYSTEM_PYTHON="${FLOW_GRPO_SYSTEM_PYTHON:-python3}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu118}"
export FLOW_GRPO_SYSTEM_PYTHON

cd "${FLOW_GRPO_PROJECT_DIR}"
bash scripts/preflight_vm.sh

if [[ ! -x .venv/bin/python ]]; then
  "${FLOW_GRPO_SYSTEM_PYTHON}" -m venv .venv
fi

source scripts/env.sh
"${FLOW_GRPO_PYTHON}" -m pip install --upgrade pip wheel setuptools
"${FLOW_GRPO_PYTHON}" -m pip install \
  torch==2.6.0 torchvision==0.21.0 \
  --index-url "${TORCH_INDEX_URL}"
"${FLOW_GRPO_PYTHON}" -m pip install -r requirements.txt
"${FLOW_GRPO_PYTHON}" -m pip check
"${FLOW_GRPO_PYTHON}" -m unittest discover -s tests -v

"${FLOW_GRPO_PYTHON}" scripts/validate_setup.py \
  --mixed-precision "${FLOW_GRPO_MIXED_PRECISION:-fp16}" \
  --minimum-free-disk-gb "${FLOW_GRPO_MINIMUM_FREE_DISK_GB:-35}"

echo "Environment is ready. Run: bash scripts/run_smoke_training.sh"
