#!/usr/bin/env bash
set -euo pipefail

FLOW_GRPO_PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FLOW_GRPO_SYSTEM_PYTHON="${FLOW_GRPO_SYSTEM_PYTHON:-python3}"
FLOW_GRPO_MINIMUM_FREE_DISK_GB="${FLOW_GRPO_MINIMUM_FREE_DISK_GB:-35}"

if ! command -v "${FLOW_GRPO_SYSTEM_PYTHON}" >/dev/null 2>&1; then
  echo "Python executable not found: ${FLOW_GRPO_SYSTEM_PYTHON}" >&2
  echo "Install Python 3.10-3.12 and its venv package before continuing." >&2
  exit 1
fi

"${FLOW_GRPO_SYSTEM_PYTHON}" -c \
  'import sys; assert (3, 10) <= sys.version_info[:2] <= (3, 12), "Python 3.10, 3.11, or 3.12 is required"'

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi was not found. Install the NVIDIA driver on the VM." >&2
  exit 1
fi

if ! nvidia-smi --query-gpu=name,memory.total,driver_version \
  --format=csv,noheader; then
  echo "The NVIDIA driver is installed but no accessible GPU was found." >&2
  exit 1
fi

free_kb="$(df -Pk "${FLOW_GRPO_PROJECT_DIR}" | awk 'NR == 2 {print $4}')"
required_kb="$((FLOW_GRPO_MINIMUM_FREE_DISK_GB * 1024 * 1024))"
if (( free_kb < required_kb )); then
  free_gb="$((free_kb / 1024 / 1024))"
  echo "Only ${free_gb} GB is free; ${FLOW_GRPO_MINIMUM_FREE_DISK_GB} GB is required." >&2
  exit 1
fi

echo "VM preflight passed."
