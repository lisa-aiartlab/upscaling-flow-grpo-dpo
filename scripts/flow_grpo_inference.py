"""Run Stable Diffusion x4 upscaling with a trained Flow-GRPO LoRA.

Example:
    python scripts/flow_grpo_inference.py \
        --image flow_grpo_dataset/lr/0001_artwork_26_like.png \
        --prompt "An eighteenth-century decorative landscape with a river"
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any


# Configure every cache before importing Hugging Face or PyTorch. The project
# lives on D:, so model downloads, CUDA kernels and temporary files stay there.
PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
CACHE_DIRECTORY = PROJECT_DIRECTORY / ".cache"
os.environ.setdefault("HF_HOME", str(CACHE_DIRECTORY / "huggingface"))
os.environ.setdefault("HF_HUB_CACHE", str(CACHE_DIRECTORY / "huggingface" / "hub"))
os.environ.setdefault("TORCH_HOME", str(CACHE_DIRECTORY / "torch"))
os.environ.setdefault("CUDA_CACHE_PATH", str(CACHE_DIRECTORY / "cuda"))
os.environ.setdefault("PIP_CACHE_DIR", str(CACHE_DIRECTORY / "pip"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIRECTORY))
os.environ.setdefault("TEMP", str(CACHE_DIRECTORY / "tmp"))
os.environ.setdefault("TMP", str(CACHE_DIRECTORY / "tmp"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

for cache_path in (
    CACHE_DIRECTORY,
    CACHE_DIRECTORY / "huggingface" / "hub",
    CACHE_DIRECTORY / "torch",
    CACHE_DIRECTORY / "cuda",
    CACHE_DIRECTORY / "pip",
    CACHE_DIRECTORY / "tmp",
):
    cache_path.mkdir(parents=True, exist_ok=True)

import torch
from PIL import Image, ImageOps
from peft import LoraConfig, set_peft_model_state_dict

from diffusers import DDIMScheduler, StableDiffusionUpscalePipeline


DEFAULT_CHECKPOINT = PROJECT_DIRECTORY / "flow_grpo_output" / "final" / "flow_grpo_lora.pt"
DEFAULT_OUTPUT = PROJECT_DIRECTORY / "inference_output" / "upscaled.png"


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_DIRECTORY / path
    return path.resolve()


def load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"LoRA checkpoint was not found: {path}")

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    required_keys = {"config", "unet_lora"}
    missing_keys = required_keys.difference(checkpoint)
    if missing_keys:
        raise ValueError(
            f"Invalid Flow-GRPO checkpoint; missing keys: {sorted(missing_keys)}"
        )

    state_dict = checkpoint["unet_lora"]
    non_finite = [
        name for name, value in state_dict.items() if not torch.isfinite(value).all()
    ]
    if non_finite:
        raise ValueError(
            f"Checkpoint contains non-finite LoRA values, first key: {non_finite[0]}"
        )
    return checkpoint


def prepare_image(path: Path, resolution: int) -> Image.Image:
    if not path.exists():
        raise FileNotFoundError(f"Input image was not found: {path}")
    with Image.open(path) as image:
        return ImageOps.fit(
            image.convert("RGB"),
            (resolution, resolution),
            method=Image.Resampling.LANCZOS,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="Input low-resolution image")
    parser.add_argument("--prompt", required=True, help="Description of the image")
    parser.add_argument("--negative-prompt", help="Elements that should not appear")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--model-id", help="Override model ID stored in the checkpoint")
    parser.add_argument("--resolution", type=int, help="LR side length; default from checkpoint")
    parser.add_argument("--inference-steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, help="Default from checkpoint")
    parser.add_argument("--noise-level", type=int, help="Default from checkpoint")
    parser.add_argument("--eta", type=float, help="DDIM stochasticity; default from checkpoint")
    parser.add_argument("--lora-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this inference script.")
    if not 0.0 <= args.lora_scale <= 2.0:
        raise ValueError("--lora-scale must be between 0 and 2.")

    checkpoint_path = resolve_path(args.checkpoint)
    input_path = resolve_path(args.image)
    output_path = resolve_path(args.output)
    checkpoint = load_checkpoint(checkpoint_path)
    training_config = checkpoint["config"]

    model_id = args.model_id or training_config.get(
        "model_id", "stabilityai/stable-diffusion-x4-upscaler"
    )
    resolution = args.resolution or int(training_config.get("resolution", 120))
    guidance_scale = (
        args.guidance_scale
        if args.guidance_scale is not None
        else float(training_config.get("guidance_scale", 7.5))
    )
    noise_level = (
        args.noise_level
        if args.noise_level is not None
        else int(training_config.get("noise_level", 20))
    )
    eta = (
        args.eta
        if args.eta is not None
        else float(training_config.get("eta", 1.0))
    )
    lora_rank = int(training_config.get("lora_rank", 4))

    if resolution % 8:
        raise ValueError("Resolution must be divisible by 8.")

    pipe = StableDiffusionUpscalePipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        local_files_only=args.local_files_only,
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.vae.enable_slicing()
    pipe.enable_attention_slicing()

    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_rank,
        init_lora_weights="gaussian",
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    )
    pipe.unet.add_adapter(lora_config, adapter_name="default")
    incompatible = set_peft_model_state_dict(
        pipe.unet,
        checkpoint["unet_lora"],
        adapter_name="default",
    )
    if incompatible.unexpected_keys:
        raise ValueError(
            "Unexpected keys while loading LoRA: "
            + ", ".join(incompatible.unexpected_keys[:5])
        )
    pipe.unet.set_adapters("default", weights=args.lora_scale)
    pipe.unet.eval()
    pipe = pipe.to("cuda")

    low_resolution_image = prepare_image(input_path, resolution)
    generator = torch.Generator(device="cuda").manual_seed(args.seed)

    with torch.inference_mode():
        result = pipe(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            image=low_resolution_image,
            num_inference_steps=args.inference_steps,
            guidance_scale=guidance_scale,
            noise_level=noise_level,
            eta=eta,
            generator=generator,
        ).images[0]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(output_path)
    print(f"Loaded Flow-GRPO LoRA: {checkpoint_path}")
    print(f"Checkpoint training step: {checkpoint.get('step', 'unknown')}")
    print(f"LoRA scale: {args.lora_scale}")
    print(f"Input: {input_path} -> {resolution}x{resolution}")
    print(f"Output: {output_path} ({result.width}x{result.height})")


if __name__ == "__main__":
    main()
