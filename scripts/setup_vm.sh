#!/usr/bin/env bash
set -euo pipefail

FLOW_GRPO_PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FLOW_GRPO_SYSTEM_PYTHON="${FLOW_GRPO_SYSTEM_PYTHON:-python3}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"

cd "${FLOW_GRPO_PROJECT_DIR}"

if ! command -v "${FLOW_GRPO_SYSTEM_PYTHON}" >/dev/null 2>&1; then
  echo "Python executable not found: ${FLOW_GRPO_SYSTEM_PYTHON}" >&2
  exit 1
fi

if [[ ! -x .venv/bin/python ]]; then
  "${FLOW_GRPO_SYSTEM_PYTHON}" -m venv .venv
fi

source scripts/env.sh
"${FLOW_GRPO_PYTHON}" -m pip install --upgrade pip wheel setuptools
"${FLOW_GRPO_PYTHON}" -m pip install \
  torch==2.4.1 torchvision==0.19.1 \
  --index-url "${TORCH_INDEX_URL}"
"${FLOW_GRPO_PYTHON}" -m pip install -r requirements.txt

"${FLOW_GRPO_PYTHON}" scripts/validate_setup.py

echo "Environment is ready. Run: bash scripts/run_smoke_training.sh"
