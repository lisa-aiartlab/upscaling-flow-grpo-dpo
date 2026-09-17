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
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from peft import LoraConfig, get_peft_model_state_dict
from torch import Tensor
from transformers import CLIPModel, CLIPProcessor

from diffusers import Flux2KleinPipeline


FLUX2_KLEIN_LORA_TARGETS = [
    "to_k",
    "to_q",
    "to_v",
    "to_out.0",
    "to_qkv_mlp_proj",
    *[f"single_transformer_blocks.{index}.attn.to_out" for index in range(24)],
]


# metrics_checker.py находится на один каталог выше текущего обучающего скрипта.
scripts_directory = Path(__file__).resolve().parents[1]
if str(scripts_directory) not in sys.path:
    sys.path.insert(0, str(scripts_directory))

from metrics_checker import ArtQualityEvaluator


# Конфигурация всех параметров генерации, награды и обучения Flow-GRPO.
@dataclass
class FlowGRPOConfig:
    model_id: str = "black-forest-labs/FLUX.2-klein-4B"
    manifest: str = "flow_grpo_dataset/upscaling_dataset/manifest.json"
    lr_pipeline: str | None = None
    output_dir: str = "flow_grpo_output"
    resolution: int = 120
    group_size: int = 4
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
    mixed_precision: str = "bf16"
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
        if config.resolution % 4:
            raise ValueError("resolution must be divisible by 4.")

        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type != "cuda":
            raise RuntimeError("Training FLUX.2 Klein requires a CUDA GPU.")

        dtype_by_name = {"fp16": torch.float16, "bf16": torch.bfloat16}
        self.weight_dtype = dtype_by_name[config.mixed_precision]
        self.generator = torch.Generator(device=self.device).manual_seed(config.seed)

        self.pipe = Flux2KleinPipeline.from_pretrained(
            config.model_id,
            torch_dtype=self.weight_dtype,
        ).to(self.device)
        self.pipe.set_progress_bar_config(disable=True)
        self.pipe.vae.enable_slicing()
        self.pipe.vae.requires_grad_(False).eval()
        self.pipe.text_encoder.requires_grad_(False).eval()
        self.pipe.transformer.requires_grad_(False)

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
        return self.pipe.encode_prompt(
            prompt=[prompt] * self.config.group_size,
            device=self.device,
            num_images_per_prompt=1,
        )

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

        return {key: value / updates for key, value in metrics.items()}

    def _save_checkpoint(self, name: str) -> Path:
        checkpoint_dir = Path(self.config.output_dir) / name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        state = {
            "step": self.global_step,
            "config": asdict(self.config),
            "transformer_lora": {
                key: value.detach().cpu()
                for key, value in get_peft_model_state_dict(
                    self.pipe.transformer
                ).items()
            },
        }
        torch.save(state, checkpoint_dir / "flow_grpo_lora.pt")
        return checkpoint_dir

    def train(self, records: list[ManifestRecord]) -> None:
        random_generator = random.Random(self.config.seed)
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(self.config.epochs):
            epoch_records = records.copy()
            random_generator.shuffle(epoch_records)

            for record in epoch_records:
                low_resolution = self._prepare_low_resolution_image(record.lr_path)
                prompt_embeddings, text_ids = self._encode_prompt(record.prompt)
                image_latents, image_latent_ids = self._prepare_condition(
                    low_resolution
                )

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
                self.global_step += 1

                reward_text = ", ".join(f"{value:.4f}" for value in rewards.tolist())
                part_means = ", ".join(
                    f"{key}={float(value.mean()):.4f}"
                    for key, value in reward_parts.items()
                )
                print(
                    f"epoch={epoch + 1} step={self.global_step} "
                    f"rewards=[{reward_text}] {part_means} "
                    f"loss={update_metrics['loss']:.6f} "
                    f"kl={update_metrics['approx_kl']:.6f} "
                    f"clipfrac={update_metrics['clip_fraction']:.4f}"
                )

                if self.global_step % self.config.save_every == 0:
                    self._save_checkpoint(f"checkpoint-{self.global_step}")

        final_path = self._save_checkpoint("final")
        print(f"Flow-GRPO training finished. LoRA checkpoint: {final_path}")


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
    parser.add_argument("--model-id", default="black-forest-labs/FLUX.2-klein-4B")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--grpo-epochs", type=int, default=2)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--inference-steps", type=int, default=4)
    parser.add_argument("--resolution", type=int, default=120)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=1.0e-5)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument("--prompt-reward-weight", type=float, default=1.0)
    parser.add_argument("--fidelity-reward-weight", type=float, default=1.0)
    parser.add_argument("--reference-reward-weight", type=float, default=0.25)
    parser.add_argument("--metrics-reward-weight", type=float, default=0.25)
    parser.add_argument("--clip-model-id", default="openai/clip-vit-base-patch32")
    parser.add_argument("--reward-device", default="cpu")
    parser.add_argument("--mixed-precision", choices=["fp16", "bf16"], default="bf16")
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    clip_model_id = None if args.clip_model_id.lower() == "none" else args.clip_model_id
    return FlowGRPOConfig(
        model_id=args.model_id,
        manifest=args.manifest,
        lr_pipeline=args.lr_pipeline,
        output_dir=args.output_dir,
        resolution=args.resolution,
        group_size=args.group_size,
        epochs=args.epochs,
        grpo_epochs=args.grpo_epochs,
        inference_steps=args.inference_steps,
        guidance_scale=args.guidance_scale,
        eta=args.eta,
        learning_rate=args.learning_rate,
        clip_epsilon=args.clip_epsilon,
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
