"""Train FLUX.2 Klein 4B for image upscaling with grouped policy optimization.

The training path implemented here is:

    low-resolution image + prompt
                  -> FLUX.2 Klein image-conditioned generation
                  -> four stochastic trajectories / images
                  -> image and prompt reward
                  -> group-relative advantages
                  -> clipped GRPO update of LoRA weights

The input may be a paired manifest or the bundled HR degradation manifest.

Example:
    python scripts/training_scripts/flow_grpo.py

Required packages:
    pip install torch torchvision diffusers transformers accelerate peft safetensors opencv-python numpy
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import signal
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict
from torch import Tensor
from transformers import CLIPModel, CLIPProcessor

from diffusers import Flux2KleinPipeline


FLUX2_KLEIN_LORA_TARGETS = (
    r"(?:transformer_blocks\.\d+\.attn\.(?:to_q|to_k|to_v|to_out\.0)"
    r"|single_transformer_blocks\.\d+\.attn\.(?:to_qkv_mlp_proj|to_out))"
)
DEFAULT_MODEL_ID = "black-forest-labs/FLUX.2-klein-4B"
# Pin model files as well as Python packages so a VM rebuild is reproducible.
DEFAULT_MODEL_REVISION = "e7b7dc27f91deacad38e78976d1f2b499d76a294"


# metrics_checker.py находится на один каталог выше текущего обучающего скрипта.
scripts_directory = Path(__file__).resolve().parents[1]
if str(scripts_directory) not in sys.path:
    sys.path.insert(0, str(scripts_directory))

from metrics_checker import ArtQualityEvaluator


# Конфигурация всех параметров генерации, награды и обучения Flow-GRPO.
@dataclass
class FlowGRPOConfig:
    model_id: str = DEFAULT_MODEL_ID
    model_revision: str = DEFAULT_MODEL_REVISION
    manifest: str = "flow_grpo_dataset/upscaling_dataset/manifest.json"
    lr_pipeline: str | None = None
    output_dir: str = "flow_grpo_output"
    resume_from_checkpoint: str | None = None
    # Hybrid (план, эксперимент В): старт Flow-GRPO с LoRA после офлайн-DPO.
    init_from_lora: str | None = None
    resolution: int = 128
    group_size: int = 2
    epochs: int = 1
    grpo_epochs: int = 2
    inference_steps: int = 4
    guidance_scale: float = 1.0
    eta: float = 1.0
    learning_rate: float = 1.0e-5
    max_grad_norm: float = 1.0
    clip_epsilon: float = 0.2
    advantage_epsilon: float = 1.0e-6
    lora_rank: int = 4
    prompt_reward_weight: float = 1.0
    fidelity_reward_weight: float = 1.0
    reference_reward_weight: float = 0.25
    metrics_reward_weight: float = 0.25
    clip_model_id: str | None = "openai/clip-vit-base-patch32"
    reward_device: str = "cpu"
    mixed_precision: str = "fp16"
    save_every: int = 25
    max_samples: int | None = None
    seed: int = 42


# Одна обучающая запись из манифеста, созданного baseline_results.py.
@dataclass
class ManifestRecord:
    lr_path: Path
    prompt: str
    reference_path: Path | None


# Результат генерации группы изображений вместе с сохранённой flow-траекторией.
@dataclass
class Rollout:
    """A detached rollout; each entry is one stochastic flow transition."""

    states: list[Tensor]
    next_states: list[Tensor]
    timesteps: list[float]
    sigmas: list[float]
    next_sigmas: list[float]
    old_log_probs: list[Tensor]
    latent_ids: Tensor
    images: Tensor


class ShutdownRequested(RuntimeError):
    """Raised at a safe point after SIGTERM or SIGINT was received."""


def _compute_empirical_mu(image_seq_len: int, num_steps: int) -> float:
    """Match the timestep shift used by the FLUX.2 Klein pipeline."""

    short_slope, short_intercept = 8.73809524e-05, 1.89833333
    long_slope, long_intercept = 0.00016927, 0.45666666
    if image_seq_len > 4300:
        return float(long_slope * image_seq_len + long_intercept)

    mu_at_200 = long_slope * image_seq_len + long_intercept
    mu_at_10 = short_slope * image_seq_len + short_intercept
    slope = (mu_at_200 - mu_at_10) / 190.0
    return float(mu_at_200 + (num_steps - 200.0) * slope)


def _resolve_manifest_path(value: str, manifest_path: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path

    candidates = [Path.cwd() / path, manifest_path.parent / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def _find_by_stem(directory: Path, filename: str) -> Path:
    """Resolve a dataset file even when a manifest contains the wrong suffix."""

    requested = directory / filename
    if requested.is_file():
        return requested.resolve()

    matches = [
        candidate
        for candidate in directory.glob(f"{Path(filename).stem}.*")
        if candidate.is_file()
    ]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Could not uniquely resolve {filename!r} in {directory}; "
            f"found {len(matches)} matches."
        )
    return matches[0].resolve()


def _load_degraded_dataset_manifest(
    raw_records: list[dict[str, Any]],
    manifest_path: Path,
    lr_pipeline: str | None,
) -> list[ManifestRecord]:
    """Expand the HR manifest plus metadata.csv into trainable LR/HR pairs."""

    metadata_path = manifest_path.parent / "metadata.csv"
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"The HR-only manifest requires degradation metadata: {metadata_path}"
        )

    with metadata_path.open("r", encoding="utf-8-sig", newline="") as file:
        metadata_rows = list(csv.DictReader(file))

    available_pipelines = sorted({row["pipeline"] for row in metadata_rows})
    if lr_pipeline and lr_pipeline not in available_pipelines:
        raise ValueError(
            f"Unknown LR pipeline {lr_pipeline!r}; choose one of "
            f"{available_pipelines}."
        )

    rows_by_stem: dict[str, list[dict[str, str]]] = {}
    for row in metadata_rows:
        if lr_pipeline and row["pipeline"] != lr_pipeline:
            continue
        rows_by_stem.setdefault(Path(row["filename"]).stem, []).append(row)

    records: list[ManifestRecord] = []
    for raw in raw_records:
        if "hr_path" not in raw or "prompt" not in raw:
            raise ValueError(
                "Each HR manifest record must contain 'hr_path' and 'prompt'."
            )

        requested_reference = Path(str(raw["hr_path"]))
        reference_directory = manifest_path.parent / requested_reference.parent
        if not reference_directory.is_dir():
            # The fetched manifest prefixes paths with "upscaling_dataset/"
            # even though it already lives inside that directory.
            reference_directory = (
                manifest_path.parent / requested_reference.parent.name
            )
        reference_path = _find_by_stem(
            reference_directory, requested_reference.name
        )
        rows = rows_by_stem.get(requested_reference.stem, [])
        if not rows:
            raise ValueError(
                f"No LR degradation records found for "
                f"{requested_reference.name!r}."
            )

        for row in rows:
            lr_path = _find_by_stem(
                manifest_path.parent / row["pipeline"], row["filename"]
            )
            records.append(
                ManifestRecord(
                    lr_path=lr_path,
                    prompt=str(raw["prompt"]),
                    reference_path=reference_path,
                )
            )
    return records


def load_baseline_manifest(
    path: str,
    max_samples: int | None,
    lr_pipeline: str | None = None,
) -> list[ManifestRecord]:
    """Read a paired manifest or expand the repository's HR degradation dataset."""

    manifest_path = Path(path).expanduser().resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Baseline manifest was not found: {manifest_path}. "
            "Create it with BaselineReferenceLogger from baseline_results.py first."
        )

    with manifest_path.open("r", encoding="utf-8") as file:
        raw_records: list[dict[str, Any]] = json.load(file)

    if raw_records and "hr_path" in raw_records[0]:
        records = _load_degraded_dataset_manifest(
            raw_records, manifest_path, lr_pipeline
        )
    else:
        records = []
        for raw in raw_records:
            if "lr_path" not in raw or "prompt" not in raw:
                raise ValueError(
                    "Each paired manifest record must contain "
                    "'lr_path' and 'prompt'."
                )
            reference_value = raw.get("pref_generated_path")
            records.append(
                ManifestRecord(
                    lr_path=_resolve_manifest_path(raw["lr_path"], manifest_path),
                    prompt=str(raw["prompt"]),
                    reference_path=(
                        _resolve_manifest_path(reference_value, manifest_path)
                        if reference_value
                        else None
                    ),
                )
            )

    if max_samples is not None:
        records = records[:max_samples]

    if not records:
        raise ValueError(f"The baseline manifest is empty: {manifest_path}")
    return records


def tensor_to_pil(image: Tensor) -> Image.Image:
    image = image.detach().float().clamp(0, 1).cpu()
    array = (image.permute(1, 2, 0).numpy() * 255.0).round().astype("uint8")
    return Image.fromarray(array)


# Вычисляет общую награду за соответствие промпту, исходному изображению и эталону.
class UpscalingReward:
    """Prompt alignment + LR consistency + proximity to the saved baseline."""

    def __init__(self, config: FlowGRPOConfig):
        self.config = config
        self.device = torch.device(config.reward_device)
        self.clip_model: CLIPModel | None = None
        self.clip_processor: CLIPProcessor | None = None
        self.metrics_evaluator = ArtQualityEvaluator(device="cpu")

        if config.clip_model_id:
            self.clip_processor = CLIPProcessor.from_pretrained(config.clip_model_id)
            self.clip_model = CLIPModel.from_pretrained(config.clip_model_id)
            self.clip_model.requires_grad_(False).eval().to(self.device)

    @torch.no_grad()
    def _prompt_scores(self, images: Tensor, prompt: str) -> Tensor:
        if self.clip_model is None or self.clip_processor is None:
            return torch.zeros(images.shape[0], dtype=torch.float32)

        pil_images = [tensor_to_pil(image) for image in images]
        inputs = self.clip_processor(
            text=[prompt] * len(pil_images),
            images=pil_images,
            return_tensors="pt",
            padding=True,
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        image_features = self.clip_model.get_image_features(pixel_values=inputs["pixel_values"])
        text_features = self.clip_model.get_text_features(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
        )
        image_features = F.normalize(image_features.float(), dim=-1)
        text_features = F.normalize(text_features.float(), dim=-1)
        return (image_features * text_features).sum(dim=-1).cpu()

    @staticmethod
    def _to_uint8_rgb(image: Tensor) -> Any:
        return (
            image.detach()
            .float()
            .clamp(0, 1)
            .mul(255)
            .round()
            .to(torch.uint8)
            .permute(1, 2, 0)
            .contiguous()
            .cpu()
            .numpy()
        )

    @torch.no_grad()
    def _metrics_checker_scores(
        self,
        images: Tensor,
        low_resolution_images: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        # Приводим обе стороны к одному размеру, как в evaluate_pair из metrics_checker.py.
        downscaled_images = F.interpolate(
            images,
            size=low_resolution_images.shape[-2:],
            mode="area",
        )
        original = self._to_uint8_rgb(low_resolution_images[0])
        quality_scores: list[float] = []
        sharpness_penalties: list[float] = []
        saturation_penalties: list[float] = []

        # ``strict=`` was added to zip in Python 3.10. These tensors are built
        # from the same batch, so their lengths are already guaranteed equal.
        for image, downscaled_image in zip(images, downscaled_images):
            generated = self._to_uint8_rgb(image)
            generated_downscaled = self._to_uint8_rgb(downscaled_image)
            sharpness = self.metrics_evaluator.compute_edge_sharpness_penalty(generated)
            saturation = self.metrics_evaluator.compute_color_saturation_shift(
                original, generated_downscaled
            )

            # Логарифм ограничивает масштаб дисперсии Лапласиана, а насыщенность
            # переводится из диапазона 0..255 в сопоставимый диапазон 0..1.
            normalized_sharpness = math.log1p(max(0.0, sharpness)) / 10.0
            normalized_saturation = max(0.0, saturation) / 255.0
            sharpness_penalties.append(normalized_sharpness)
            saturation_penalties.append(normalized_saturation)
            quality_scores.append(-(normalized_sharpness + normalized_saturation))

        return (
            torch.tensor(quality_scores, dtype=torch.float32),
            torch.tensor(sharpness_penalties, dtype=torch.float32),
            torch.tensor(saturation_penalties, dtype=torch.float32),
        )

    @staticmethod
    def _load_reference(path: Path, size: tuple[int, int]) -> Tensor:
        with Image.open(path) as image:
            image = image.convert("RGB").resize(size, Image.Resampling.LANCZOS)
            data = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
            data = data.reshape(image.height, image.width, 3)
        return data.permute(2, 0, 1).float().div(255.0)

    @torch.no_grad()
    def __call__(
        self,
        images: Tensor,
        low_resolution_image: Tensor,
        prompt: str,
        reference_path: Path | None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        images_cpu = images.detach().float().cpu()
        prompt_scores = self._prompt_scores(images_cpu, prompt)

        low_resolution_01 = low_resolution_image.detach().float().cpu().add(1).div(2)
        low_resolution_01 = low_resolution_01.expand(images_cpu.shape[0], -1, -1, -1)
        downscaled = F.interpolate(
            images_cpu,
            size=low_resolution_01.shape[-2:],
            mode="area",
        )
        fidelity_scores = 1.0 - (downscaled - low_resolution_01).square().mean((1, 2, 3))
        metrics_scores, sharpness_penalties, saturation_penalties = (
            self._metrics_checker_scores(images_cpu, low_resolution_01)
        )

        reference_scores = torch.zeros_like(fidelity_scores)
        if reference_path is not None and reference_path.exists():
            width, height = images_cpu.shape[-1], images_cpu.shape[-2]
            reference = self._load_reference(reference_path, (width, height)).unsqueeze(0)
            reference_scores = 1.0 - (images_cpu - reference).square().mean((1, 2, 3))

        rewards = (
            self.config.prompt_reward_weight * prompt_scores
            + self.config.fidelity_reward_weight * fidelity_scores
            + self.config.reference_reward_weight * reference_scores
            + self.config.metrics_reward_weight * metrics_scores
        )
        parts = {
            "prompt": prompt_scores,
            "fidelity": fidelity_scores,
            "reference": reference_scores,
            "metrics": metrics_scores,
            "sharpness_penalty": sharpness_penalties,
            "saturation_penalty": saturation_penalties,
        }
        return rewards, parts


# Управляет генерацией четырёх вариантов и обновлением обучаемых LoRA-параметров.
class FlowGRPOTrainer:
    def __init__(self, config: FlowGRPOConfig):
        if config.group_size < 2:
            raise ValueError("GRPO needs at least two outputs in each group.")
        if config.eta <= 0:
            raise ValueError("eta must be positive so that rollout transitions are stochastic.")
        if config.resolution % 16:
            raise ValueError("resolution must be divisible by 16 for FLUX.2 latent packing.")
        if config.epochs <= 0 or config.grpo_epochs <= 0:
            raise ValueError("epochs and grpo_epochs must be positive.")
        if config.inference_steps < 2:
            raise ValueError("inference_steps must be at least 2.")
        if config.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if config.lora_rank <= 0:
            raise ValueError("lora_rank must be positive.")
        if config.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive.")
        if config.advantage_epsilon <= 0:
            raise ValueError("advantage_epsilon must be positive.")
        if config.save_every <= 0:
            raise ValueError("save_every must be positive.")

        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type != "cuda":
            raise RuntimeError("Training FLUX.2 Klein requires a CUDA GPU.")
        properties = torch.cuda.get_device_properties(self.device)
        device_arch = f"sm_{properties.major}{properties.minor}"
        compiled_arches = torch.cuda.get_arch_list()
        if compiled_arches and device_arch not in compiled_arches:
            raise RuntimeError(
                f"The installed PyTorch wheel does not contain {device_arch} kernels "
                f"for {properties.name}; compiled architectures: {compiled_arches}."
            )
        if config.mixed_precision == "bf16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError(
                f"{properties.name} does not support bf16. Restart with "
                "--mixed-precision fp16 (required for NVIDIA V100/Volta)."
            )

        dtype_by_name = {"fp16": torch.float16, "bf16": torch.bfloat16}
        self.weight_dtype = dtype_by_name[config.mixed_precision]
        self.generator = torch.Generator(device=self.device).manual_seed(config.seed)

        self.pipe = Flux2KleinPipeline.from_pretrained(
            config.model_id,
            revision=config.model_revision,
            torch_dtype=self.weight_dtype,
        ).to(self.device)
        self.pipe.set_progress_bar_config(disable=True)
        self.pipe.vae.enable_slicing()
        self.pipe.vae.requires_grad_(False).eval()
        self.pipe.text_encoder.requires_grad_(False).eval()
        self.pipe.transformer.requires_grad_(False)

        matched_lora_modules = [
            name
            for name, module in self.pipe.transformer.named_modules()
            if re.fullmatch(FLUX2_KLEIN_LORA_TARGETS, name)
            and isinstance(module, torch.nn.Linear)
        ]
        if not matched_lora_modules:
            raise RuntimeError(
                "No FLUX.2 transformer modules matched the LoRA target pattern. "
                "Check the pinned Diffusers/model versions."
            )

        lora_config = LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_rank,
            init_lora_weights="gaussian",
            target_modules=FLUX2_KLEIN_LORA_TARGETS,
        )
        self.pipe.transformer.add_adapter(lora_config)
        self.pipe.transformer.enable_gradient_checkpointing()
        for parameter in self.pipe.transformer.parameters():
            if parameter.requires_grad:
                parameter.data = parameter.data.float()

        self.trainable_parameters = [
            parameter
            for parameter in self.pipe.transformer.parameters()
            if parameter.requires_grad
        ]
        if not self.trainable_parameters:
            raise RuntimeError("No trainable LoRA parameters were created.")

        trainable_count = sum(parameter.numel() for parameter in self.trainable_parameters)
        print(
            f"LoRA: {len(matched_lora_modules)} modules, "
            f"{trainable_count:,} trainable parameters",
            flush=True,
        )

        self.optimizer = torch.optim.AdamW(
            self.trainable_parameters,
            lr=config.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=1.0e-2,
        )
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.weight_dtype == torch.float16)
        self.scheduler = self.pipe.scheduler
        self.reward = UpscalingReward(config)
        self.global_step = 0
        self.next_epoch = 0
        self.next_record_index = 0
        self.record_count: int | None = None
        self.resume_record_count: int | None = None
        self.shutdown_requested = False
        self.shutdown_signal: int | None = None
        self.prompt_cache: dict[str, tuple[Tensor, Tensor]] = {}

        if config.init_from_lora:
            self._load_lora_weights_only(config.init_from_lora)
        if config.resume_from_checkpoint:
            self._load_checkpoint(config.resume_from_checkpoint)

    def _handle_shutdown_signal(self, signum: int, _frame: Any) -> None:
        """Request a checkpoint without doing unsafe I/O inside the handler."""

        if not self.shutdown_requested:
            self.shutdown_requested = True
            self.shutdown_signal = signum
            signal_name = signal.Signals(signum).name
            print(
                f"Received {signal_name}; stopping at the nearest safe point "
                "and saving a shutdown checkpoint.",
                file=sys.stderr,
                flush=True,
            )

    def _raise_if_shutdown_requested(self) -> None:
        if self.shutdown_requested:
            raise ShutdownRequested

    @staticmethod
    def _checkpoint_file(path: str | Path) -> Path:
        checkpoint_path = Path(path).expanduser()
        if checkpoint_path.is_dir():
            for name in ("flow_grpo_lora.pt", "dpo_lora.pt"):
                candidate = checkpoint_path / name
                if candidate.is_file():
                    return candidate.resolve()
            checkpoint_path = checkpoint_path / "flow_grpo_lora.pt"
        return checkpoint_path.resolve()

    def _load_lora_weights_only(self, path: str | Path) -> None:
        """Load transformer_lora from a DPO or Flow-GRPO checkpoint (Hybrid warm-start)."""

        if self.config.resume_from_checkpoint:
            raise ValueError(
                "Use either --init-from-lora or --resume-from-checkpoint, not both."
            )

        checkpoint_path = self._checkpoint_file(path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Init LoRA checkpoint was not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if "transformer_lora" not in checkpoint:
            raise ValueError(f"Checkpoint has no transformer_lora: {checkpoint_path}")

        saved_config = checkpoint.get("config") or {}
        for key in ("model_id", "model_revision", "lora_rank", "mixed_precision"):
            if key not in saved_config:
                continue
            saved_value = saved_config[key]
            current_value = getattr(self.config, key)
            if saved_value != current_value:
                raise ValueError(
                    f"Init LoRA checkpoint has {key}={saved_value!r}, but the current "
                    f"run requested {current_value!r}."
                )

        incompatible = set_peft_model_state_dict(
            self.pipe.transformer,
            checkpoint["transformer_lora"],
            adapter_name="default",
        )
        if incompatible.unexpected_keys:
            raise ValueError(
                "Unexpected LoRA keys in init checkpoint: "
                + ", ".join(incompatible.unexpected_keys[:5])
            )
        algorithm = checkpoint.get("algorithm", "unknown")
        print(
            f"Initialized LoRA from {checkpoint_path} "
            f"(source={algorithm}, step={checkpoint.get('step', 'unknown')})",
            flush=True,
        )

    def _load_checkpoint(self, path: str | Path) -> None:
        checkpoint_path = self._checkpoint_file(path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint was not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        required_keys = {
            "config",
            "transformer_lora",
            "optimizer",
            "scaler",
            "progress",
            "rng_state",
        }
        missing_keys = required_keys.difference(checkpoint)
        if missing_keys:
            raise ValueError(
                "Checkpoint cannot be resumed exactly; missing keys: "
                f"{sorted(missing_keys)}. Legacy LoRA-only checkpoints can still "
                "be used for inference."
            )

        saved_config = checkpoint["config"]
        for key in (
            "model_id",
            "model_revision",
            "lora_rank",
            "mixed_precision",
            "manifest",
            "lr_pipeline",
            "max_samples",
            "seed",
        ):
            saved_value = saved_config.get(
                key,
                DEFAULT_MODEL_REVISION if key == "model_revision" else None,
            )
            current_value = getattr(self.config, key)
            if saved_value != current_value:
                raise ValueError(
                    f"Resume checkpoint has {key}={saved_value!r}, but the current "
                    f"run requested {current_value!r}."
                )

        incompatible = set_peft_model_state_dict(
            self.pipe.transformer,
            checkpoint["transformer_lora"],
            adapter_name="default",
        )
        if incompatible.unexpected_keys:
            raise ValueError(
                "Unexpected LoRA keys in resume checkpoint: "
                + ", ".join(incompatible.unexpected_keys[:5])
            )

        self.optimizer.load_state_dict(checkpoint["optimizer"])
        # Allow an explicitly supplied learning rate to be used after resuming.
        for parameter_group in self.optimizer.param_groups:
            parameter_group["lr"] = self.config.learning_rate
        self.scaler.load_state_dict(checkpoint["scaler"])

        progress = checkpoint["progress"]
        self.global_step = int(progress["global_step"])
        self.next_epoch = int(progress["next_epoch"])
        self.next_record_index = int(progress["next_record_index"])
        saved_record_count = progress.get("record_count")
        self.resume_record_count = (
            int(saved_record_count) if saved_record_count is not None else None
        )

        rng_state = checkpoint["rng_state"]
        random.setstate(rng_state["python"])
        np.random.set_state(rng_state["numpy"])
        torch.set_rng_state(rng_state["torch"])
        self.generator.set_state(rng_state["generator"])
        for device_index, cuda_state in enumerate(rng_state.get("cuda", [])):
            if device_index >= torch.cuda.device_count():
                break
            torch.cuda.set_rng_state(cuda_state, device=device_index)

        print(
            f"Resumed checkpoint {checkpoint_path}: step={self.global_step}, "
            f"next_epoch={self.next_epoch + 1}, "
            f"next_record={self.next_record_index + 1}",
            flush=True,
        )

    def _prepare_low_resolution_image(self, path: Path) -> Tensor:
        if not path.exists():
            raise FileNotFoundError(f"Low-resolution image was not found: {path}")
        with Image.open(path) as image:
            image = ImageOps.fit(
                image.convert("RGB"),
                (self.config.resolution, self.config.resolution),
                method=Image.Resampling.LANCZOS,
            )
            image_tensor = self.pipe.image_processor.preprocess(image)
        return image_tensor.to(device=self.device, dtype=self.weight_dtype)

    @torch.no_grad()
    def _encode_prompt(self, prompt: str) -> tuple[Tensor, Tensor]:
        if prompt not in self.prompt_cache:
            raise RuntimeError(
                "Prompt embeddings were not cached before training started."
            )
        prompt_embeddings, text_ids = self.prompt_cache[prompt]
        return (
            prompt_embeddings.to(self.device).repeat(self.config.group_size, 1, 1),
            text_ids.to(self.device).repeat(self.config.group_size, 1, 1),
        )

    @torch.no_grad()
    def _cache_prompt_embeddings(self, records: list[ManifestRecord]) -> None:
        """Encode every distinct prompt once, then free text-encoder VRAM."""

        unique_prompts = list(dict.fromkeys(record.prompt for record in records))
        print(f"Caching {len(unique_prompts)} unique prompt embeddings...", flush=True)
        self.pipe.text_encoder.eval().to(self.device)
        for prompt in unique_prompts:
            self._raise_if_shutdown_requested()
            prompt_embeddings, text_ids = self.pipe.encode_prompt(
                prompt=prompt,
                device=self.device,
                num_images_per_prompt=1,
            )
            self.prompt_cache[prompt] = (
                prompt_embeddings.detach().cpu(),
                text_ids.detach().cpu(),
            )

        self.pipe.text_encoder.to("cpu")
        torch.cuda.empty_cache()
        print("Prompt cache ready; text encoder moved to CPU.", flush=True)

    @torch.no_grad()
    def _prepare_condition(self, image: Tensor) -> tuple[Tensor, Tensor]:
        return self.pipe.prepare_image_latents(
            images=[image],
            batch_size=self.config.group_size,
            generator=self.generator,
            device=self.device,
            dtype=self.pipe.vae.dtype,
        )

    def _predict_velocity(
        self,
        latents: Tensor,
        timestep: float,
        latent_ids: Tensor,
        image_latents: Tensor,
        image_latent_ids: Tensor,
        prompt_embeddings: Tensor,
        text_ids: Tensor,
    ) -> Tensor:
        timestep_tensor = torch.full(
            (latents.shape[0],),
            timestep / 1000.0,
            device=self.device,
            dtype=latents.dtype,
        )
        model_input = torch.cat([latents, image_latents], dim=1).to(
            self.pipe.transformer.dtype
        )
        model_image_ids = torch.cat([latent_ids, image_latent_ids], dim=1)
        prediction = self.pipe.transformer(
            hidden_states=model_input,
            timestep=timestep_tensor,
            guidance=None,
            encoder_hidden_states=prompt_embeddings,
            txt_ids=text_ids,
            img_ids=model_image_ids,
            joint_attention_kwargs=None,
            return_dict=False,
        )[0]
        return prediction[:, : latents.shape[1]]

    def _flow_mean_and_std(
        self,
        model_output: Tensor,
        sigma: float,
        next_sigma: float,
        sample: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Return the stochastic Euler-ancestral flow transition parameters."""

        output_dtype = sample.dtype
        sample = sample.float()
        model_output = model_output.float()
        sigma_tensor = torch.as_tensor(sigma, device=sample.device, dtype=torch.float32)
        next_sigma_tensor = torch.as_tensor(
            next_sigma, device=sample.device, dtype=torch.float32
        )

        if next_sigma <= 0.0:
            sigma_up = torch.zeros_like(next_sigma_tensor)
        else:
            variance = (
                next_sigma_tensor.square()
                * (sigma_tensor.square() - next_sigma_tensor.square()).clamp_min(0)
                / sigma_tensor.square().clamp_min(1.0e-12)
            )
            sigma_up = torch.minimum(
                next_sigma_tensor,
                self.config.eta * variance.sqrt(),
            )
        sigma_down = (next_sigma_tensor.square() - sigma_up.square()).clamp_min(0).sqrt()
        mean = sample + (sigma_down - sigma_tensor) * model_output
        return mean.to(output_dtype), sigma_up.to(output_dtype)

    @staticmethod
    def _transition_log_probability(value: Tensor, mean: Tensor, std: Tensor) -> Tensor:
        variance = std.square().clamp_min(1.0e-12)
        log_probability = -0.5 * (
            (value - mean).square() / variance + torch.log(2 * math.pi * variance)
        )
        return log_probability.flatten(1).mean(dim=1)

    @torch.no_grad()
    def _decode_latents(
        self,
        latents: Tensor,
        latent_ids: Tensor,
        height: int,
        width: int,
    ) -> Tensor:
        latent_height = 2 * (height // (self.pipe.vae_scale_factor * 2))
        latent_width = 2 * (width // (self.pipe.vae_scale_factor * 2))
        latents = self.pipe._unpack_latents_with_ids(
            latents,
            latent_ids,
            latent_height // 2,
            latent_width // 2,
        )
        batch_norm_mean = self.pipe.vae.bn.running_mean.view(1, -1, 1, 1).to(
            latents.device, latents.dtype
        )
        batch_norm_std = torch.sqrt(
            self.pipe.vae.bn.running_var.view(1, -1, 1, 1)
            + self.pipe.vae.config.batch_norm_eps
        ).to(latents.device, latents.dtype)
        latents = self.pipe._unpatchify_latents(
            latents * batch_norm_std + batch_norm_mean
        )
        decoded = self.pipe.vae.decode(latents, return_dict=False)[0]
        if not torch.isfinite(decoded).all():
            raise FloatingPointError("VAE decoder produced non-finite values.")
        images = self.pipe.image_processor.postprocess(decoded, output_type="pt")
        return images.float().cpu()

    @torch.no_grad()
    def _rollout(
        self,
        image_latents: Tensor,
        image_latent_ids: Tensor,
        prompt_embeddings: Tensor,
        text_ids: Tensor,
    ) -> Rollout:
        self.pipe.transformer.eval()
        target_resolution = self.config.resolution * 4
        latent_channels = self.pipe.transformer.config.in_channels // 4
        latents, latent_ids = self.pipe.prepare_latents(
            batch_size=self.config.group_size,
            num_latents_channels=latent_channels,
            height=target_resolution,
            width=target_resolution,
            dtype=prompt_embeddings.dtype,
            device=self.device,
            generator=self.generator,
        )

        image_seq_len = latents.shape[1]
        step_count = self.config.inference_steps
        mu = _compute_empirical_mu(image_seq_len, step_count)
        if getattr(self.scheduler.config, "use_flow_sigmas", False):
            self.scheduler.set_timesteps(
                step_count,
                device=self.device,
                mu=mu,
            )
        else:
            flow_sigmas = np.linspace(1.0, 1.0 / step_count, step_count)
            self.scheduler.set_timesteps(
                sigmas=flow_sigmas,
                device=self.device,
                mu=mu,
            )
        timesteps = [float(value) for value in self.scheduler.timesteps]
        sigmas = [float(value) for value in self.scheduler.sigmas]
        if len(sigmas) != len(timesteps) + 1:
            raise RuntimeError(
                "FLUX scheduler must provide one more sigma than timestep."
            )

        states: list[Tensor] = []
        next_states: list[Tensor] = []
        stored_timesteps: list[float] = []
        stored_sigmas: list[float] = []
        stored_next_sigmas: list[float] = []
        old_log_probs: list[Tensor] = []

        for index, timestep in enumerate(timesteps):
            self._raise_if_shutdown_requested()
            sigma = sigmas[index]
            next_sigma = sigmas[index + 1]
            model_output = self._predict_velocity(
                latents,
                timestep,
                latent_ids,
                image_latents,
                image_latent_ids,
                prompt_embeddings,
                text_ids,
            )
            self._raise_if_shutdown_requested()
            if not torch.isfinite(model_output).all():
                raise FloatingPointError(
                    f"FLUX transformer produced non-finite values at timestep {timestep}."
                )
            mean, std = self._flow_mean_and_std(
                model_output, sigma, next_sigma, latents
            )
            if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
                raise FloatingPointError(
                    f"Flow transition produced non-finite values at timestep {timestep}."
                )

            if float(std) > 0:
                noise = torch.randn(
                    latents.shape,
                    generator=self.generator,
                    device=self.device,
                    dtype=latents.dtype,
                )
                next_latents = mean + std * noise
                states.append(latents.cpu())
                next_states.append(next_latents.cpu())
                stored_timesteps.append(timestep)
                stored_sigmas.append(sigma)
                stored_next_sigmas.append(next_sigma)
                old_log_probs.append(
                    self._transition_log_probability(next_latents, mean, std).float().cpu()
                )
            else:
                next_latents = mean
            latents = next_latents

        images = self._decode_latents(
            latents,
            latent_ids,
            target_resolution,
            target_resolution,
        )
        self._raise_if_shutdown_requested()
        return Rollout(
            states=states,
            next_states=next_states,
            timesteps=stored_timesteps,
            sigmas=stored_sigmas,
            next_sigmas=stored_next_sigmas,
            old_log_probs=old_log_probs,
            latent_ids=latent_ids.cpu(),
            images=images,
        )

    def _grpo_update(
        self,
        rollout: Rollout,
        advantages: Tensor,
        image_latents: Tensor,
        image_latent_ids: Tensor,
        prompt_embeddings: Tensor,
        text_ids: Tensor,
    ) -> dict[str, float]:
        if not rollout.states:
            raise RuntimeError("No stochastic transitions were produced for the GRPO update.")

        metrics = {"loss": 0.0, "approx_kl": 0.0, "clip_fraction": 0.0}
        updates = 0
        self.pipe.transformer.train()
        latent_ids = rollout.latent_ids.to(self.device)

        for _ in range(self.config.grpo_epochs):
            if self.shutdown_requested:
                if updates == 0:
                    raise ShutdownRequested
                break
            self.optimizer.zero_grad(set_to_none=True)
            epoch_loss = 0.0
            epoch_kl = 0.0
            epoch_clip_fraction = 0.0

            for state_cpu, next_state_cpu, timestep, sigma, next_sigma, old_log_prob_cpu in zip(
                rollout.states,
                rollout.next_states,
                rollout.timesteps,
                rollout.sigmas,
                rollout.next_sigmas,
                rollout.old_log_probs,
            ):
                if self.shutdown_requested:
                    self.optimizer.zero_grad(set_to_none=True)
                    if updates == 0:
                        raise ShutdownRequested
                    break
                state = state_cpu.to(self.device, dtype=self.weight_dtype)
                next_state = next_state_cpu.to(self.device, dtype=self.weight_dtype)
                old_log_probability = old_log_prob_cpu.to(self.device)

                with torch.autocast(
                    device_type="cuda", dtype=self.weight_dtype, enabled=True
                ):
                    model_output = self._predict_velocity(
                        state,
                        timestep,
                        latent_ids,
                        image_latents,
                        image_latent_ids,
                        prompt_embeddings,
                        text_ids,
                    )
                    mean, std = self._flow_mean_and_std(
                        model_output, sigma, next_sigma, state
                    )
                    new_log_probability = self._transition_log_probability(
                        next_state, mean, std
                    ).float()
                    log_ratio = (new_log_probability - old_log_probability).clamp(-20, 20)
                    ratio = log_ratio.exp()
                    unclipped = ratio * advantages
                    clipped = ratio.clamp(
                        1 - self.config.clip_epsilon,
                        1 + self.config.clip_epsilon,
                    ) * advantages
                    loss = -torch.minimum(unclipped, clipped).mean()
                    loss = loss / len(rollout.states)

                self.scaler.scale(loss).backward()
                epoch_loss += float(loss.detach())
                epoch_kl += float(((ratio - 1) - log_ratio).mean().detach())
                epoch_clip_fraction += float(
                    ((ratio - 1).abs() > self.config.clip_epsilon).float().mean().detach()
                )

            if self.shutdown_requested:
                self.optimizer.zero_grad(set_to_none=True)
                if updates == 0:
                    raise ShutdownRequested
                break

            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.trainable_parameters, self.config.max_grad_norm
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()

            transition_count = len(rollout.states)
            metrics["loss"] += epoch_loss
            metrics["approx_kl"] += epoch_kl / transition_count
            metrics["clip_fraction"] += epoch_clip_fraction / transition_count
            updates += 1

            # An optimizer step is a commit point. If SIGTERM arrived inside
            # optimizer.step(), keep this update and finish the current record
            # with fewer GRPO epochs instead of applying it twice after resume.
            if self.shutdown_requested:
                break

        return {key: value / updates for key, value in metrics.items()}

    def _save_checkpoint(self, name: str, reason: str = "periodic") -> Path:
        checkpoint_dir = Path(self.config.output_dir) / name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        state = {
            "checkpoint_version": 2,
            "step": self.global_step,
            "config": asdict(self.config),
            "reason": reason,
            "transformer_lora": {
                key: value.detach().cpu()
                for key, value in get_peft_model_state_dict(
                    self.pipe.transformer
                ).items()
            },
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "progress": {
                "global_step": self.global_step,
                "next_epoch": self.next_epoch,
                "next_record_index": self.next_record_index,
                "record_count": self.record_count,
            },
            "rng_state": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all(),
                "generator": self.generator.get_state(),
            },
        }
        checkpoint_path = checkpoint_dir / "flow_grpo_lora.pt"
        temporary_path = checkpoint_dir / "flow_grpo_lora.pt.tmp"
        torch.save(state, temporary_path)
        # Keep the previous checkpoint intact if the VM is killed during write.
        os.replace(temporary_path, checkpoint_path)
        return checkpoint_dir

    def train(self, records: list[ManifestRecord]) -> None:
        self.record_count = len(records)
        if (
            self.resume_record_count is not None
            and self.resume_record_count != self.record_count
        ):
            raise ValueError(
                "The resume checkpoint was created with "
                f"{self.resume_record_count} records, but the current manifest "
                f"produced {self.record_count}."
            )
        if self.next_epoch < 0 or self.next_record_index < 0:
            raise ValueError("Resume checkpoint contains negative progress indices.")
        if self.next_record_index > self.record_count:
            raise ValueError(
                "Resume checkpoint record index is outside the current dataset."
            )

        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        handled_signals = [signal.SIGTERM, signal.SIGINT]
        previous_handlers = {
            handled_signal: signal.getsignal(handled_signal)
            for handled_signal in handled_signals
        }
        for handled_signal in handled_signals:
            signal.signal(handled_signal, self._handle_shutdown_signal)

        try:
            self._cache_prompt_embeddings(records)
            for epoch in range(self.next_epoch, self.config.epochs):
                epoch_records = records.copy()
                # This order can be reconstructed from epoch and seed, so a
                # checkpoint only needs the next record index.
                random.Random(self.config.seed + epoch).shuffle(epoch_records)
                start_index = self.next_record_index if epoch == self.next_epoch else 0

                for record_index in range(start_index, len(epoch_records)):
                    self._raise_if_shutdown_requested()
                    record = epoch_records[record_index]
                    low_resolution = self._prepare_low_resolution_image(record.lr_path)
                    prompt_embeddings, text_ids = self._encode_prompt(record.prompt)
                    self._raise_if_shutdown_requested()
                    image_latents, image_latent_ids = self._prepare_condition(
                        low_resolution
                    )
                    self._raise_if_shutdown_requested()

                    rollout = self._rollout(
                        image_latents,
                        image_latent_ids,
                        prompt_embeddings,
                        text_ids,
                    )
                    rewards, reward_parts = self.reward(
                        rollout.images,
                        low_resolution,
                        record.prompt,
                        record.reference_path,
                    )
                    self._raise_if_shutdown_requested()
                    advantages = (rewards - rewards.mean()) / (
                        rewards.std(unbiased=False) + self.config.advantage_epsilon
                    )
                    advantages = advantages.to(self.device)

                    update_metrics = self._grpo_update(
                        rollout,
                        advantages,
                        image_latents,
                        image_latent_ids,
                        prompt_embeddings,
                        text_ids,
                    )

                    # Commit progress only after at least one optimizer step.
                    self.global_step += 1
                    next_record_index = record_index + 1
                    if next_record_index == len(epoch_records):
                        self.next_epoch = epoch + 1
                        self.next_record_index = 0
                    else:
                        self.next_epoch = epoch
                        self.next_record_index = next_record_index

                    reward_text = ", ".join(
                        f"{value:.4f}" for value in rewards.tolist()
                    )
                    part_means = ", ".join(
                        f"{key}={float(value.mean()):.4f}"
                        for key, value in reward_parts.items()
                    )
                    print(
                        f"epoch={epoch + 1} step={self.global_step} "
                        f"rewards=[{reward_text}] {part_means} "
                        f"loss={update_metrics['loss']:.6f} "
                        f"kl={update_metrics['approx_kl']:.6f} "
                        f"clipfrac={update_metrics['clip_fraction']:.4f}",
                        flush=True,
                    )

                    if self.global_step % self.config.save_every == 0:
                        self._save_checkpoint(
                            f"checkpoint-{self.global_step}", reason="periodic"
                        )
                    self._raise_if_shutdown_requested()

            final_path = self._save_checkpoint("final", reason="completed")
            print(
                f"Flow-GRPO training finished. LoRA checkpoint: {final_path}",
                flush=True,
            )
        except ShutdownRequested:
            signal_name = (
                signal.Signals(self.shutdown_signal).name
                if self.shutdown_signal is not None
                else "shutdown request"
            )
            shutdown_path = self._save_checkpoint(
                "shutdown", reason=f"received {signal_name}"
            )
            print(
                f"Training stopped safely. Resume checkpoint: {shutdown_path}",
                flush=True,
            )
        finally:
            for handled_signal, previous_handler in previous_handlers.items():
                signal.signal(handled_signal, previous_handler)


def parse_args() -> FlowGRPOConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default="flow_grpo_dataset/upscaling_dataset/manifest.json",
    )
    parser.add_argument(
        "--lr-pipeline",
        help=(
            "For an HR-only manifest, train on one degradation pipeline "
            "instead of all."
        ),
    )
    parser.add_argument("--output-dir", default="flow_grpo_output")
    parser.add_argument(
        "--resume-from-checkpoint",
        help=(
            "Checkpoint file or directory created by this trainer. Restores "
            "LoRA, optimizer, scaler, RNG state, epoch, and dataset position."
        ),
    )
    parser.add_argument(
        "--init-from-lora",
        help=(
            "Load only transformer_lora from a DPO or Flow-GRPO checkpoint, "
            "then start a fresh Flow-GRPO optimizer (Hybrid: DPO → Flow-GRPO)."
        ),
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument(
        "--model-revision",
        default=DEFAULT_MODEL_REVISION,
        help="Hugging Face commit/tag for reproducible base-model loading.",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--grpo-epochs", type=int, default=2)
    parser.add_argument("--group-size", type=int, default=2)
    parser.add_argument("--inference-steps", type=int, default=4)
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=1.0e-5)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--advantage-epsilon", type=float, default=1.0e-6)
    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument("--prompt-reward-weight", type=float, default=1.0)
    parser.add_argument("--fidelity-reward-weight", type=float, default=1.0)
    parser.add_argument("--reference-reward-weight", type=float, default=0.25)
    parser.add_argument("--metrics-reward-weight", type=float, default=0.25)
    parser.add_argument("--clip-model-id", default="openai/clip-vit-base-patch32")
    parser.add_argument("--reward-device", default="cpu")
    parser.add_argument("--mixed-precision", choices=["fp16", "bf16"], default="fp16")
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    clip_model_id = None if args.clip_model_id.lower() == "none" else args.clip_model_id
    return FlowGRPOConfig(
        model_id=args.model_id,
        model_revision=args.model_revision,
        manifest=args.manifest,
        lr_pipeline=args.lr_pipeline,
        output_dir=args.output_dir,
        resume_from_checkpoint=args.resume_from_checkpoint,
        init_from_lora=args.init_from_lora,
        resolution=args.resolution,
        group_size=args.group_size,
        epochs=args.epochs,
        grpo_epochs=args.grpo_epochs,
        inference_steps=args.inference_steps,
        guidance_scale=args.guidance_scale,
        eta=args.eta,
        learning_rate=args.learning_rate,
        max_grad_norm=args.max_grad_norm,
        clip_epsilon=args.clip_epsilon,
        advantage_epsilon=args.advantage_epsilon,
        lora_rank=args.lora_rank,
        prompt_reward_weight=args.prompt_reward_weight,
        fidelity_reward_weight=args.fidelity_reward_weight,
        reference_reward_weight=args.reference_reward_weight,
        metrics_reward_weight=args.metrics_reward_weight,
        clip_model_id=clip_model_id,
        reward_device=args.reward_device,
        mixed_precision=args.mixed_precision,
        save_every=args.save_every,
        max_samples=args.max_samples,
        seed=args.seed,
    )


def main() -> None:
    config = parse_args()
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    records = load_baseline_manifest(
        config.manifest, config.max_samples, config.lr_pipeline
    )
    trainer = FlowGRPOTrainer(config)
    trainer.train(records)


if __name__ == "__main__":
    main()
