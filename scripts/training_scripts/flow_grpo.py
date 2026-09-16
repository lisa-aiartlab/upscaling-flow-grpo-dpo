"""Train the Stable Diffusion x4 upscaler with grouped policy optimization.

The training path implemented here is:

    low-resolution image + prompt
                  -> x4 upscaler
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

import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from peft import LoraConfig, get_peft_model_state_dict
from torch import Tensor
from transformers import CLIPModel, CLIPProcessor

from diffusers import DDIMScheduler, StableDiffusionUpscalePipeline


# metrics_checker.py находится на один каталог выше текущего обучающего скрипта.
scripts_directory = Path(__file__).resolve().parents[1]
if str(scripts_directory) not in sys.path:
    sys.path.insert(0, str(scripts_directory))

from metrics_checker import ArtQualityEvaluator


# Конфигурация всех параметров генерации, награды и обучения Flow-GRPO.
@dataclass
class FlowGRPOConfig:
    model_id: str = "stabilityai/stable-diffusion-x4-upscaler"
    manifest: str = "flow_grpo_dataset/upscaling_dataset/manifest.json"
    lr_pipeline: str | None = None
    output_dir: str = "flow_grpo_output"
    resolution: int = 120
    group_size: int = 4
    epochs: int = 1
    grpo_epochs: int = 2
    inference_steps: int = 20
    noise_level: int = 20
    guidance_scale: float = 7.5
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


# Результат генерации группы изображений вместе с сохранённой DDIM-траекторией.
@dataclass
class Rollout:
    """A detached rollout; each list entry is one stochastic DDIM transition."""

    states: list[Tensor]
    next_states: list[Tensor]
    timesteps: list[int]
    previous_timesteps: list[int]
    old_log_probs: list[Tensor]
    images: Tensor


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
        if config.resolution % 8:
            raise ValueError("resolution must be divisible by 8.")

        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type != "cuda":
            raise RuntimeError("Training the x4 upscaler requires a CUDA GPU.")

        dtype_by_name = {"fp16": torch.float16, "bf16": torch.bfloat16}
        self.weight_dtype = dtype_by_name[config.mixed_precision]
        self.generator = torch.Generator(device=self.device).manual_seed(config.seed)

        self.pipe = StableDiffusionUpscalePipeline.from_pretrained(
            config.model_id,
            torch_dtype=self.weight_dtype,
        ).to(self.device)
        self.pipe.set_progress_bar_config(disable=True)
        self.pipe.vae.enable_slicing()
        self.pipe.enable_attention_slicing()
        self.pipe.vae.requires_grad_(False).eval()
        self.pipe.text_encoder.requires_grad_(False).eval()
        self.pipe.unet.requires_grad_(False)

        lora_config = LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_rank,
            init_lora_weights="gaussian",
            target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        )
        self.pipe.unet.add_adapter(lora_config)
        self.pipe.unet.enable_gradient_checkpointing()
        for parameter in self.pipe.unet.parameters():
            if parameter.requires_grad:
                parameter.data = parameter.data.float()

        self.trainable_parameters = [
            parameter for parameter in self.pipe.unet.parameters() if parameter.requires_grad
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
        self.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
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
    def _encode_prompt(self, prompt: str) -> Tensor:
        tokenizer = self.pipe.tokenizer
        text_encoder = self.pipe.text_encoder

        def encode(text: str) -> Tensor:
            tokens = tokenizer(
                [text],
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            return text_encoder(tokens.input_ids.to(self.device))[0]

        conditional = encode(prompt).repeat(self.config.group_size, 1, 1)
        unconditional = encode("").repeat(self.config.group_size, 1, 1)
        return torch.cat([unconditional, conditional], dim=0)

    def _prepare_condition(self, image: Tensor) -> tuple[Tensor, Tensor]:
        noise_level = torch.tensor(
            [self.config.noise_level], device=self.device, dtype=torch.long
        )
        noise = torch.randn(
            image.shape,
            generator=self.generator,
            device=self.device,
            dtype=image.dtype,
        )
        noisy_image = self.pipe.low_res_scheduler.add_noise(image, noise, noise_level)
        noisy_image = noisy_image.repeat(self.config.group_size, 1, 1, 1)
        class_labels = noise_level.repeat(self.config.group_size)
        return noisy_image, class_labels

    def _predict_noise(
        self,
        latents: Tensor,
        timestep: int,
        image_condition: Tensor,
        class_labels: Tensor,
        prompt_embeddings: Tensor,
    ) -> Tensor:
        timestep_tensor = torch.tensor(timestep, device=self.device, dtype=torch.long)
        latent_input = torch.cat([latents, latents], dim=0)
        latent_input = self.scheduler.scale_model_input(latent_input, timestep_tensor)
        model_input = torch.cat(
            [latent_input, torch.cat([image_condition, image_condition], dim=0)], dim=1
        )
        labels = torch.cat([class_labels, class_labels], dim=0)
        prediction = self.pipe.unet(
            model_input,
            timestep_tensor,
            encoder_hidden_states=prompt_embeddings,
            class_labels=labels,
            return_dict=False,
        )[0]
        unconditional, conditional = prediction.chunk(2)
        return unconditional + self.config.guidance_scale * (conditional - unconditional)

    def _ddim_mean_and_std(
        self,
        model_output: Tensor,
        timestep: int,
        previous_timestep: int,
        sample: Tensor,
    ) -> tuple[Tensor, Tensor]:
        # Scheduler coefficients become numerically unstable in fp16 near the
        # final denoising timestep (alpha is close to 1 and beta is close to
        # zero). Keep the inexpensive DDIM scalar/tensor arithmetic in fp32;
        # gradients still flow back through the cast to the fp16 UNet output.
        output_dtype = sample.dtype
        sample = sample.float()
        model_output = model_output.float()
        alphas = self.scheduler.alphas_cumprod
        alpha_t = alphas[timestep].to(device=sample.device, dtype=torch.float32)
        if previous_timestep >= 0:
            alpha_previous = alphas[previous_timestep].to(
                device=sample.device, dtype=torch.float32
            )
        else:
            alpha_previous = self.scheduler.final_alpha_cumprod.to(
                device=sample.device, dtype=torch.float32
            )

        beta_t = 1 - alpha_t
        prediction_type = self.scheduler.config.prediction_type
        if prediction_type == "epsilon":
            predicted_original = (sample - beta_t.sqrt() * model_output) / alpha_t.sqrt()
            predicted_epsilon = model_output
        elif prediction_type == "sample":
            predicted_original = model_output
            predicted_epsilon = (sample - alpha_t.sqrt() * predicted_original) / beta_t.sqrt()
        elif prediction_type == "v_prediction":
            predicted_original = alpha_t.sqrt() * sample - beta_t.sqrt() * model_output
            predicted_epsilon = alpha_t.sqrt() * model_output + beta_t.sqrt() * sample
        else:
            raise ValueError(f"Unsupported prediction type: {prediction_type}")

        if self.scheduler.config.thresholding:
            predicted_original = self.scheduler._threshold_sample(predicted_original)
        elif self.scheduler.config.clip_sample:
            clip_range = self.scheduler.config.clip_sample_range
            predicted_original = predicted_original.clamp(-clip_range, clip_range)

        beta_previous = 1 - alpha_previous
        variance = (beta_previous / beta_t) * (1 - alpha_t / alpha_previous)
        variance = variance.clamp_min(0)
        standard_deviation = self.config.eta * variance.sqrt()
        direction_scale = (1 - alpha_previous - standard_deviation.square()).clamp_min(0).sqrt()
        mean = alpha_previous.sqrt() * predicted_original + direction_scale * predicted_epsilon
        return mean.to(output_dtype), standard_deviation.to(output_dtype)

    @staticmethod
    def _transition_log_probability(value: Tensor, mean: Tensor, std: Tensor) -> Tensor:
        variance = std.square().clamp_min(1.0e-12)
        log_probability = -0.5 * (
            (value - mean).square() / variance + torch.log(2 * math.pi * variance)
        )
        return log_probability.flatten(1).mean(dim=1)

    @torch.no_grad()
    def _rollout(
        self,
        image_condition: Tensor,
        class_labels: Tensor,
        prompt_embeddings: Tensor,
    ) -> Rollout:
        self.pipe.unet.eval()
        self.scheduler.set_timesteps(self.config.inference_steps, device=self.device)
        timesteps = [int(value) for value in self.scheduler.timesteps]

        latent_channels = self.pipe.unet.config.in_channels - image_condition.shape[1]
        shape = (
            self.config.group_size,
            latent_channels,
            image_condition.shape[-2],
            image_condition.shape[-1],
        )
        latents = torch.randn(
            shape,
            generator=self.generator,
            device=self.device,
            dtype=self.weight_dtype,
        )
        latents = latents * self.scheduler.init_noise_sigma

        states: list[Tensor] = []
        next_states: list[Tensor] = []
        stored_timesteps: list[int] = []
        stored_previous_timesteps: list[int] = []
        old_log_probs: list[Tensor] = []

        for index, timestep in enumerate(timesteps):
            previous_timestep = timesteps[index + 1] if index + 1 < len(timesteps) else -1
            model_output = self._predict_noise(
                latents,
                timestep,
                image_condition,
                class_labels,
                prompt_embeddings,
            )
            if not torch.isfinite(model_output).all():
                raise FloatingPointError(
                    f"UNet produced non-finite values at timestep {timestep}."
                )
            mean, std = self._ddim_mean_and_std(
                model_output, timestep, previous_timestep, latents
            )
            if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
                raise FloatingPointError(
                    f"DDIM transition produced non-finite values at timestep {timestep}."
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
                stored_previous_timesteps.append(previous_timestep)
                old_log_probs.append(
                    self._transition_log_probability(next_latents, mean, std).float().cpu()
                )
            else:
                next_latents = mean
            latents = next_latents

        needs_upcasting = (
            self.pipe.vae.dtype == torch.float16
            and getattr(self.pipe.vae.config, "force_upcast", False)
        )
        if needs_upcasting:
            self.pipe.vae.to(dtype=torch.float32)
            latents = latents.float()
        decoded = self.pipe.vae.decode(
            latents / self.pipe.vae.config.scaling_factor, return_dict=False
        )[0]
        if not torch.isfinite(decoded).all():
            raise FloatingPointError("VAE decoder produced non-finite values.")
        if needs_upcasting:
            self.pipe.vae.to(dtype=self.weight_dtype)
        images = decoded.add(1).div(2).clamp(0, 1).float().cpu()
        return Rollout(
            states=states,
            next_states=next_states,
            timesteps=stored_timesteps,
            previous_timesteps=stored_previous_timesteps,
            old_log_probs=old_log_probs,
            images=images,
        )

    def _grpo_update(
        self,
        rollout: Rollout,
        advantages: Tensor,
        image_condition: Tensor,
        class_labels: Tensor,
        prompt_embeddings: Tensor,
    ) -> dict[str, float]:
        if not rollout.states:
            raise RuntimeError("No stochastic transitions were produced for the GRPO update.")

        metrics = {"loss": 0.0, "approx_kl": 0.0, "clip_fraction": 0.0}
        updates = 0
        self.pipe.unet.train()

        for _ in range(self.config.grpo_epochs):
            self.optimizer.zero_grad(set_to_none=True)
            epoch_loss = 0.0
            epoch_kl = 0.0
            epoch_clip_fraction = 0.0

            for state_cpu, next_state_cpu, timestep, previous_timestep, old_log_prob_cpu in zip(
                rollout.states,
                rollout.next_states,
                rollout.timesteps,
                rollout.previous_timesteps,
                rollout.old_log_probs,
            ):
                state = state_cpu.to(self.device, dtype=self.weight_dtype)
                next_state = next_state_cpu.to(self.device, dtype=self.weight_dtype)
                old_log_probability = old_log_prob_cpu.to(self.device)

                with torch.autocast(
                    device_type="cuda", dtype=self.weight_dtype, enabled=True
                ):
                    model_output = self._predict_noise(
                        state,
                        timestep,
                        image_condition,
                        class_labels,
                        prompt_embeddings,
                    )
                    mean, std = self._ddim_mean_and_std(
                        model_output, timestep, previous_timestep, state
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
            "unet_lora": {
                key: value.detach().cpu()
                for key, value in get_peft_model_state_dict(self.pipe.unet).items()
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
                prompt_embeddings = self._encode_prompt(record.prompt)
                image_condition, class_labels = self._prepare_condition(low_resolution)

                rollout = self._rollout(
                    image_condition, class_labels, prompt_embeddings
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
                    image_condition,
                    class_labels,
                    prompt_embeddings,
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
    parser.add_argument("--model-id", default="stabilityai/stable-diffusion-x4-upscaler")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--grpo-epochs", type=int, default=2)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--inference-steps", type=int, default=20)
    parser.add_argument("--resolution", type=int, default=120)
    parser.add_argument("--noise-level", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
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
    parser.add_argument("--mixed-precision", choices=["fp16", "bf16"], default="fp16")
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
        noise_level=args.noise_level,
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
