"""Validate dependencies, CUDA, and both bundled Flow-GRPO datasets."""

from __future__ import annotations

import argparse
import inspect
import shutil
import sys
from importlib import metadata
from pathlib import Path


PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
TRAINING_DIRECTORY = PROJECT_DIRECTORY / "scripts" / "training_scripts"
sys.path.insert(0, str(TRAINING_DIRECTORY))

import torch
from PIL import Image
from diffusers import Flux2KleinPipeline

from flow_grpo import load_baseline_manifest
from dpo import load_preference_manifest


EXPECTED_VERSIONS = {
    "torch": "2.6.0",
    "torchvision": "0.21.0",
    "diffusers": "0.40.0",
    "transformers": "5.15.1",
    "accelerate": "1.14.0",
    "peft": "0.20.0",
    "safetensors": "0.8.0",
    "numpy": "1.26.4",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allow-no-cuda",
        action="store_true",
        help="Validate files and imports on a CPU host without requiring a GPU.",
    )
    parser.add_argument(
        "--mixed-precision",
        choices=["fp16", "bf16"],
        default="fp16",
        help="Precision that the training launcher will use.",
    )
    parser.add_argument(
        "--minimum-free-disk-gb",
        type=float,
        default=35.0,
        help="Required free space for model cache and checkpoints.",
    )
    return parser.parse_args()


def validate_records(name: str, records: list) -> None:
    checked_images: set[Path] = set()
    for record in records:
        if not record.lr_path.is_file():
            raise FileNotFoundError(record.lr_path)
        reference_path = getattr(record, "reference_path", None)
        chosen_path = getattr(record, "chosen_path", None)
        rejected_path = getattr(record, "rejected_path", None)
        for image_path in (reference_path, chosen_path, rejected_path):
            if image_path is not None and not image_path.is_file():
                raise FileNotFoundError(image_path)
        checked_images.add(record.lr_path)
        for image_path in (reference_path, chosen_path, rejected_path):
            if image_path is not None:
                checked_images.add(image_path)

    for image_path in checked_images:
        with Image.open(image_path) as image:
            image.verify()
    print(f"{name}: {len(records)} trainable pairs")


def validate_versions() -> None:
    for package, expected in EXPECTED_VERSIONS.items():
        installed = metadata.version(package)
        public_version = installed.split("+", maxsplit=1)[0]
        if public_version != expected:
            raise RuntimeError(
                f"{package}=={expected} is required, found {installed}. "
                "Re-run scripts/setup_vm."
            )
        print(f"{package}: {installed}")


def validate_pipeline_api() -> None:
    required_methods = {
        "encode_prompt",
        "prepare_image_latents",
        "prepare_latents",
        "_encode_vae_image",
        "_pack_latents",
        "_prepare_latent_ids",
        "_unpack_latents_with_ids",
        "_unpatchify_latents",
    }
    missing_methods = [
        name for name in required_methods if not hasattr(Flux2KleinPipeline, name)
    ]
    if missing_methods:
        raise RuntimeError(
            f"Pinned Diffusers lacks required FLUX.2 APIs: {missing_methods}"
        )

    transformer_parameters = inspect.signature(
        Flux2KleinPipeline.prepare_latents
    ).parameters
    for parameter in ("batch_size", "height", "width", "generator"):
        if parameter not in transformer_parameters:
            raise RuntimeError(
                f"Flux2KleinPipeline.prepare_latents lacks {parameter!r}."
            )
    print("FLUX.2 pipeline API: compatible")


def main() -> None:
    args = parse_args()
    if not (3, 10) <= sys.version_info[:2] <= (3, 12):
        raise RuntimeError("Python 3.10, 3.11, or 3.12 is required.")

    validate_versions()
    validate_pipeline_api()

    free_disk_gb = shutil.disk_usage(PROJECT_DIRECTORY).free / (1024**3)
    if free_disk_gb < args.minimum_free_disk_gb:
        raise RuntimeError(
            f"Only {free_disk_gb:.1f} GB is free; at least "
            f"{args.minimum_free_disk_gb:.1f} GB is required."
        )
    print(f"Free disk: {free_disk_gb:.1f} GB")

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
    validate_records(
        "DPO preference loader (curated)",
        load_preference_manifest(str(paired_manifest), None),
    )
    validate_records(
        "DPO preference loader (degradation, one pipeline)",
        load_preference_manifest(
            str(degraded_manifest),
            max_samples=2,
            lr_pipeline="LR_01_resize",
        ),
    )

    training_scripts = (
        TRAINING_DIRECTORY / "flow_grpo.py",
        TRAINING_DIRECTORY / "dpo.py",
    )
    for script_path in training_scripts:
        if not script_path.is_file():
            raise FileNotFoundError(f"Missing training script: {script_path}")
    print("Training scripts: flow_grpo.py, dpo.py")

    if not torch.cuda.is_available():
        if not args.allow_no_cuda:
            raise RuntimeError(
                "CUDA is unavailable. Install an NVIDIA driver and a matching "
                "CUDA-enabled PyTorch wheel."
            )
        print("CUDA: unavailable (allowed by --allow-no-cuda)")
    else:
        properties = torch.cuda.get_device_properties(0)
        total_vram_gb = properties.total_memory / (1024**3)
        print(
            f"CUDA: {properties.name}, capability "
            f"{properties.major}.{properties.minor}, {total_vram_gb:.1f} GB VRAM"
        )
        device_arch = f"sm_{properties.major}{properties.minor}"
        compiled_arches = torch.cuda.get_arch_list()
        if compiled_arches and device_arch not in compiled_arches:
            raise RuntimeError(
                f"The installed PyTorch wheel lacks {device_arch} kernels for "
                f"{properties.name}; compiled architectures: {compiled_arches}."
            )
        print(f"PyTorch CUDA architectures: {', '.join(compiled_arches)}")
        if args.mixed_precision == "bf16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError(
                f"{properties.name} does not support bf16. Use "
                "--mixed-precision fp16; this is mandatory for NVIDIA V100/Volta."
            )
        if total_vram_gb < 20:
            print(
                "WARNING: less than 20 GB VRAM; even group-size 2 may run out "
                "of memory with FLUX.2 Klein."
            )
    print("Setup validation passed.")


if __name__ == "__main__":
    main()
