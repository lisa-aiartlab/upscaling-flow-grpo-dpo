"""Validate dependencies, CUDA, and both bundled Flow-GRPO datasets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
TRAINING_DIRECTORY = PROJECT_DIRECTORY / "scripts" / "training_scripts"
sys.path.insert(0, str(TRAINING_DIRECTORY))

import torch

from flow_grpo import load_baseline_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allow-no-cuda",
        action="store_true",
        help="Validate files and imports on a CPU host without requiring a GPU.",
    )
    return parser.parse_args()


def validate_records(name: str, records: list) -> None:
    for record in records:
        if not record.lr_path.is_file():
            raise FileNotFoundError(record.lr_path)
        if record.reference_path is not None and not record.reference_path.is_file():
            raise FileNotFoundError(record.reference_path)
    print(f"{name}: {len(records)} trainable pairs")


def main() -> None:
    args = parse_args()
    if sys.version_info < (3, 8):
        raise RuntimeError("Python 3.8 or newer is required.")

    paired_manifest = PROJECT_DIRECTORY / "flow_grpo_dataset" / "manifest.json"
    degraded_manifest = (
        PROJECT_DIRECTORY
        / "flow_grpo_dataset"
        / "upscaling_dataset"
        / "manifest.json"
    )

    validate_records(
        "Curated preference dataset",
        load_baseline_manifest(str(paired_manifest), None),
    )
    validate_records(
        "Fetched degradation dataset",
        load_baseline_manifest(str(degraded_manifest), None),
    )

    if not torch.cuda.is_available():
        if not args.allow_no_cuda:
            raise RuntimeError(
                "CUDA is unavailable. Install an NVIDIA driver and a matching "
                "CUDA-enabled PyTorch wheel."
            )
        print("CUDA: unavailable (allowed by --allow-no-cuda)")
    else:
        print(f"CUDA: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    print("Setup validation passed.")


if __name__ == "__main__":
    main()
