"""Train FLUX.2 Klein 4B for image upscaling with offline Flow-DPO.

This is Experiment A from the research plan (DPO-like Upscale / Сокомандница 3):
offline Direct Preference Optimization on fixed preference pairs
(y_w preferred, y_l rejected) conditioned on the low-resolution image x_lr.

Training path:

    preference pair (y_w, y_l) + x_lr + prompt
                  -> encode both images to FLUX packed latents
                  -> sample a shared flow-matching noise level σ
                  -> velocity MSE under LoRA policy π_θ and frozen ref π_ref
                  -> Diffusion-/Flow-DPO logistic loss with temperature β
                  -> LoRA update that raises likelihood of y_w relative to y_l
                     without drifting too far from p_ref

The resulting LoRA checkpoint is Hybrid-ready: Flow-GRPO can later continue
from the saved ``transformer_lora`` weights (DPO → Flow-GRPO).

Example:
    python scripts/training_scripts/dpo.py \\
        --manifest flow_grpo_dataset/manifest.json \\
        --output-dir dpo_output

Preference manifest fields (any alias works):
    lr_path, prompt,
    chosen_path | y_w | preferred_path | pref_generated_path,
    rejected_path | y_l | disliked_path   (optional; see --synthetic-rejected)

Required packages:
    pip install torch torchvision diffusers transformers accelerate peft safetensors opencv-python numpy
"""

from __future__ import annotations

import argparse
import csv
import json
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

from diffusers import Flux2KleinPipeline


FLUX2_KLEIN_LORA_TARGETS = (
    r"(?:transformer_blocks\.\d+\.attn\.(?:to_q|to_k|to_v|to_out\.0)"
    r"|single_transformer_blocks\.\d+\.attn\.(?:to_qkv_mlp_proj|to_out))"
)
DEFAULT_MODEL_ID = "black-forest-labs/FLUX.2-klein-4B"
DEFAULT_MODEL_REVISION = "e7b7dc27f91deacad38e78976d1f2b499d76a294"

CHOSEN_KEYS = (
    "chosen_path",
    "y_w",
    "preferred_path",
    "pref_generated_path",
    "hr_path",
)
REJECTED_KEYS = ("rejected_path", "y_l", "disliked_path")


# ---------------------------------------------------------------------------
# Шаг 0. Конфигурация: все гиперпараметры DPO в одном месте.
# ---------------------------------------------------------------------------
@dataclass
class DPOConfig:
    model_id: str = DEFAULT_MODEL_ID
    model_revision: str = DEFAULT_MODEL_REVISION
    manifest: str = "flow_grpo_dataset/manifest.json"
    lr_pipeline: str | None = None
    output_dir: str = "dpo_output"
    resume_from_checkpoint: str | None = None
    init_from_lora: str | None = None
    # resolution — сторона x_lr; целевые y_w / y_l берутся как resolution * 4.
    resolution: int = 128
    epochs: int = 1
    learning_rate: float = 1.0e-5
    max_grad_norm: float = 1.0
    # β из плана: сила KL-отталкивания от p_ref внутри DPO-лосca.
    beta: float = 500.0
    lora_rank: int = 4
    # Если в манифесте нет y_l — синтезируем «замыленный» апскейл из x_lr.
    synthetic_rejected: str = "lanczos"
    mixed_precision: str = "fp16"
    save_every: int = 25
    max_samples: int | None = None
    seed: int = 42


@dataclass
class PreferenceRecord:
    """Одна офлайн-пара предпочтений: (x_lr, prompt, y_w, y_l)."""

    lr_path: Path
    prompt: str
    chosen_path: Path
    rejected_path: Path | None


class ShutdownRequested(RuntimeError):
    """Raised at a safe point after SIGTERM or SIGINT was received."""


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


def _first_present(raw: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = raw.get(key)
        if value:
            return str(value)
    return None


# ---------------------------------------------------------------------------
# Шаг 1. Загрузка preference-датасета.
# План, этап 2 / шаг 3: пары (y_w, y_l) для одного и того же x_lr.
# ---------------------------------------------------------------------------
def _load_degraded_preference_manifest(
    raw_records: list[dict[str, Any]],
    manifest_path: Path,
    lr_pipeline: str | None,
) -> list[PreferenceRecord]:
    """HR-манифест + metadata.csv → пары (HR = y_w, LR-pipeline sample → x_lr).

    y_l здесь ещё нет: его синтезирует тренер из x_lr (замыленный апскейл),
    что соответствует критерию плана «чёткое / естественное vs замыленное».
    """

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

    records: list[PreferenceRecord] = []
    for raw in raw_records:
        if "hr_path" not in raw or "prompt" not in raw:
            raise ValueError(
                "Each HR manifest record must contain 'hr_path' and 'prompt'."
            )

        requested_reference = Path(str(raw["hr_path"]))
        reference_directory = manifest_path.parent / requested_reference.parent
        if not reference_directory.is_dir():
            reference_directory = (
                manifest_path.parent / requested_reference.parent.name
            )
        chosen_path = _find_by_stem(
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
            rejected_value = _first_present(raw, REJECTED_KEYS)
            records.append(
                PreferenceRecord(
                    lr_path=lr_path,
                    prompt=str(raw["prompt"]),
                    chosen_path=chosen_path,
                    rejected_path=(
                        _resolve_manifest_path(rejected_value, manifest_path)
                        if rejected_value
                        else None
                    ),
                )
            )
    return records


def load_preference_manifest(
    path: str,
    max_samples: int | None,
    lr_pipeline: str | None = None,
) -> list[PreferenceRecord]:
    """Читает явные preference-пары или разворачивает HR-/curated-манифесты."""

    manifest_path = Path(path).expanduser().resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Preference manifest was not found: {manifest_path}."
        )

    with manifest_path.open("r", encoding="utf-8") as file:
        raw_records: list[dict[str, Any]] = json.load(file)

    if raw_records and "hr_path" in raw_records[0] and "lr_path" not in raw_records[0]:
        records = _load_degraded_preference_manifest(
            raw_records, manifest_path, lr_pipeline
        )
    else:
        records = []
        for raw in raw_records:
            if "lr_path" not in raw or "prompt" not in raw:
                raise ValueError(
                    "Each preference record must contain 'lr_path' and 'prompt'."
                )
            chosen_value = _first_present(raw, CHOSEN_KEYS)
            if chosen_value is None:
                raise ValueError(
                    "Each preference record needs a chosen image path "
                    f"(one of {CHOSEN_KEYS})."
                )
            rejected_value = _first_present(raw, REJECTED_KEYS)
            records.append(
                PreferenceRecord(
                    lr_path=_resolve_manifest_path(raw["lr_path"], manifest_path),
                    prompt=str(raw["prompt"]),
                    chosen_path=_resolve_manifest_path(chosen_value, manifest_path),
                    rejected_path=(
                        _resolve_manifest_path(rejected_value, manifest_path)
                        if rejected_value
                        else None
                    ),
                )
            )

    if max_samples is not None:
        records = records[:max_samples]

    if not records:
        raise ValueError(f"The preference manifest is empty: {manifest_path}")
    return records


# ---------------------------------------------------------------------------
# Шаг 2. Трейнер Flow-DPO.
# ---------------------------------------------------------------------------
class FlowDPOTrainer:
    def __init__(self, config: DPOConfig):
        if config.resolution % 16:
            raise ValueError("resolution must be divisible by 16 for FLUX.2 latent packing.")
        if config.epochs <= 0:
            raise ValueError("epochs must be positive.")
        if config.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if config.lora_rank <= 0:
            raise ValueError("lora_rank must be positive.")
        if config.beta <= 0:
            raise ValueError("beta must be positive.")
        if config.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive.")
        if config.save_every <= 0:
            raise ValueError("save_every must be positive.")
        if config.synthetic_rejected not in {"lanczos", "bicubic", "nearest"}:
            raise ValueError(
                "synthetic_rejected must be one of: lanczos, bicubic, nearest."
            )

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

        # Шаг 2.1. Базовая Flow Matching модель = p_ref до появления LoRA.
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

        # Шаг 2.2. LoRA поверх transformer: обучаем только адаптер π_θ.
        # Замороженный base через disable_adapter() остаётся π_ref ≈ p_ref.
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
            f"{trainable_count:,} trainable parameters; DPO beta={config.beta}",
            flush=True,
        )

        self.optimizer = torch.optim.AdamW(
            self.trainable_parameters,
            lr=config.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=1.0e-2,
        )
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.weight_dtype == torch.float16)
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
            for name in ("dpo_lora.pt", "flow_grpo_lora.pt"):
                candidate = checkpoint_path / name
                if candidate.is_file():
                    return candidate.resolve()
            checkpoint_path = checkpoint_path / "dpo_lora.pt"
        return checkpoint_path.resolve()

    def _load_lora_weights_only(self, path: str | Path) -> None:
        """Инициализация LoRA из предыдущего DPO/Flow-GRPO чекпоинта (Hybrid)."""

        checkpoint_path = self._checkpoint_file(path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Init LoRA checkpoint was not found: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if "transformer_lora" not in checkpoint:
            raise ValueError(f"Checkpoint has no transformer_lora: {checkpoint_path}")
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
        print(f"Initialized LoRA from {checkpoint_path}", flush=True)

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
                f"{sorted(missing_keys)}. Use --init-from-lora for LoRA-only init."
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
            "beta",
            "synthetic_rejected",
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

    # ------------------------------------------------------------------
    # Шаг 3. Подготовка изображений и текстовых эмбеддингов.
    # ------------------------------------------------------------------
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

    def _prepare_target_image(self, path: Path) -> Tensor:
        """Загружает y_w или y_l в целевом разрешении resolution*4."""

        if not path.exists():
            raise FileNotFoundError(f"Preference image was not found: {path}")
        target_resolution = self.config.resolution * 4
        with Image.open(path) as image:
            image = ImageOps.fit(
                image.convert("RGB"),
                (target_resolution, target_resolution),
                method=Image.Resampling.LANCZOS,
            )
            image_tensor = self.pipe.image_processor.preprocess(image)
        return image_tensor.to(device=self.device, dtype=self.weight_dtype)

    def _synthesize_rejected_from_lr(self, low_resolution: Tensor) -> Tensor:
        """Синтетический y_l: замыленный апскейл x_lr без генеративной модели.

        По плану непредпочтительный вариант — замыливание / потеря деталей.
        Это даёт стартовый preference-сигнал, пока нет размеченных dislike-пар.
        """

        target_resolution = self.config.resolution * 4
        resample = {
            "lanczos": Image.Resampling.LANCZOS,
            "bicubic": Image.Resampling.BICUBIC,
            "nearest": Image.Resampling.NEAREST,
        }[self.config.synthetic_rejected]

        image_01 = low_resolution.detach().float().cpu().add(1).div(2).clamp(0, 1)[0]
        array = (
            image_01.mul(255).round().to(torch.uint8).permute(1, 2, 0).numpy()
        )
        pil = Image.fromarray(array).resize(
            (target_resolution, target_resolution),
            resample=resample,
        )
        image_tensor = self.pipe.image_processor.preprocess(pil)
        return image_tensor.to(device=self.device, dtype=self.weight_dtype)

    @torch.no_grad()
    def _cache_prompt_embeddings(self, records: list[PreferenceRecord]) -> None:
        """Шаг 3.1. Один раз кодируем уникальные промпты и освобождаем text encoder."""

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

    def _encode_prompt(self, prompt: str, batch_size: int) -> tuple[Tensor, Tensor]:
        if prompt not in self.prompt_cache:
            raise RuntimeError(
                "Prompt embeddings were not cached before training started."
            )
        prompt_embeddings, text_ids = self.prompt_cache[prompt]
        return (
            prompt_embeddings.to(self.device).repeat(batch_size, 1, 1),
            text_ids.to(self.device).repeat(batch_size, 1, 1),
        )

    @torch.no_grad()
    def _prepare_condition(self, image: Tensor, batch_size: int) -> tuple[Tensor, Tensor]:
        """Шаг 3.2. Conditioning-латенты от x_lr (как в Flow-GRPO / Upscaling)."""

        return self.pipe.prepare_image_latents(
            images=[image],
            batch_size=batch_size,
            generator=self.generator,
            device=self.device,
            dtype=self.pipe.vae.dtype,
        )

    @torch.no_grad()
    def _encode_target_latents(self, image: Tensor) -> tuple[Tensor, Tensor]:
        """Шаг 3.3. Кодирование y_w / y_l в packed FLUX.2 латенты.

        Используем тот же _encode_vae_image, что и prepare_image_latents:
        VAE encode → patchify → batch-norm normalize → pack.
        """

        packed_spatial = self.pipe._encode_vae_image(image=image, generator=self.generator)
        latent_ids = self.pipe._prepare_latent_ids(packed_spatial).to(self.device)
        latents = self.pipe._pack_latents(packed_spatial)
        return latents, latent_ids

    # ------------------------------------------------------------------
    # Шаг 4. Forward velocity и Flow-DPO loss.
    # ------------------------------------------------------------------
    def _predict_velocity(
        self,
        latents: Tensor,
        timestep: Tensor,
        latent_ids: Tensor,
        image_latents: Tensor,
        image_latent_ids: Tensor,
        prompt_embeddings: Tensor,
        text_ids: Tensor,
    ) -> Tensor:
        model_input = torch.cat([latents, image_latents], dim=1).to(
            self.pipe.transformer.dtype
        )
        model_image_ids = torch.cat([latent_ids, image_latent_ids], dim=1)
        prediction = self.pipe.transformer(
            hidden_states=model_input,
            timestep=timestep,
            guidance=None,
            encoder_hidden_states=prompt_embeddings,
            txt_ids=text_ids,
            img_ids=model_image_ids,
            joint_attention_kwargs=None,
            return_dict=False,
        )[0]
        return prediction[:, : latents.shape[1]]

    def _dpo_step(
        self,
        chosen_image: Tensor,
        rejected_image: Tensor,
        low_resolution: Tensor,
        prompt: str,
    ) -> dict[str, float]:
        """Один офлайн-шаг Diffusion-/Flow-DPO на паре (y_w, y_l).

        Лосс (Wallace et al. / Flow-matching variant):

            L_θ(y) = ||v_θ(x_t, t | x_lr) - (ε - y)||²
            Δ_θ    = L_θ(y_w) - L_θ(y_l)
            Δ_ref  = L_ref(y_w) - L_ref(y_l)
            L_DPO  = -log σ( -β/2 · (Δ_θ - Δ_ref) )

        β держит π_θ рядом с p_ref (план, этап 4 / абляции KL).
        """

        batch_size = 2
        prompt_embeddings, text_ids = self._encode_prompt(prompt, batch_size)
        image_latents, image_latent_ids = self._prepare_condition(
            low_resolution, batch_size
        )

        # 4.1. Чистые латенты победителя и проигравшего.
        chosen_latents, chosen_ids = self._encode_target_latents(chosen_image)
        rejected_latents, rejected_ids = self._encode_target_latents(rejected_image)
        clean_latents = torch.cat([chosen_latents, rejected_latents], dim=0)
        # IDs совпадают по геометрии; берём chosen и дублируем на пару.
        latent_ids = chosen_ids.repeat(batch_size, 1, 1)
        del rejected_ids

        # 4.2. Общий уровень шума σ для честного сравнения пары.
        # Rectified flow: x_t = (1-σ)·x_0 + σ·ε, target velocity = ε - x_0.
        sigma = torch.rand(
            (batch_size, 1, 1),
            generator=self.generator,
            device=self.device,
            dtype=torch.float32,
        ).clamp(1.0e-4, 1.0)
        # Одинаковый σ у y_w и y_l.
        sigma = sigma[:1].expand(batch_size, -1, -1)
        noise = torch.randn(
            clean_latents.shape,
            generator=self.generator,
            device=self.device,
            dtype=clean_latents.dtype,
        )
        sigma_cast = sigma.to(dtype=clean_latents.dtype)
        noisy_latents = (1.0 - sigma_cast) * clean_latents + sigma_cast * noise
        velocity_target = noise.float() - clean_latents.float()
        timestep = (sigma.view(batch_size) * 1000.0).to(dtype=noisy_latents.dtype)

        self.pipe.transformer.train()
        self.optimizer.zero_grad(set_to_none=True)

        # 4.3. Velocity policy π_θ (с LoRA).
        with torch.autocast(
            device_type="cuda", dtype=self.weight_dtype, enabled=True
        ):
            policy_velocity = self._predict_velocity(
                noisy_latents,
                timestep,
                latent_ids,
                image_latents,
                image_latent_ids,
                prompt_embeddings,
                text_ids,
            )
            policy_losses = (policy_velocity.float() - velocity_target).square().mean(
                dim=(1, 2)
            )
            policy_chosen, policy_rejected = policy_losses[0], policy_losses[1]
            policy_diff = policy_chosen - policy_rejected

        # 4.4. Velocity reference π_ref: тот же forward без LoRA-адаптера.
        with torch.no_grad(), self.pipe.transformer.disable_adapter():
            with torch.autocast(
                device_type="cuda", dtype=self.weight_dtype, enabled=True
            ):
                reference_velocity = self._predict_velocity(
                    noisy_latents,
                    timestep,
                    latent_ids,
                    image_latents,
                    image_latent_ids,
                    prompt_embeddings,
                    text_ids,
                )
                reference_losses = (
                    (reference_velocity.float() - velocity_target).square().mean(dim=(1, 2))
                )
                reference_diff = reference_losses[0] - reference_losses[1]

        # 4.5. Logistic DPO: увеличиваем margin (Δ_ref - Δ_θ) с температурой β.
        inside_term = -0.5 * self.config.beta * (policy_diff - reference_diff)
        loss = -F.logsigmoid(inside_term)
        # Implicit reward margin для логов: насколько π_θ предпочитает y_w сильнее, чем π_ref.
        implied_reward = 0.5 * self.config.beta * (reference_diff - policy_diff)

        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(
            self.trainable_parameters, self.config.max_grad_norm
        )
        self.scaler.step(self.optimizer)
        self.scaler.update()

        accuracy = float((implied_reward > 0).float().detach())
        return {
            "loss": float(loss.detach()),
            "policy_chosen": float(policy_chosen.detach()),
            "policy_rejected": float(policy_rejected.detach()),
            "reward_margin": float(implied_reward.detach()),
            "accuracy": accuracy,
            "sigma": float(sigma.view(-1)[0].detach()),
        }

    def _save_checkpoint(self, name: str, reason: str = "periodic") -> Path:
        checkpoint_dir = Path(self.config.output_dir) / name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        state = {
            "checkpoint_version": 1,
            "algorithm": "flow_dpo",
            "step": self.global_step,
            "config": asdict(self.config),
            "reason": reason,
            # Тот же ключ, что у Flow-GRPO — удобно для Hybrid DPO → GRPO.
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
        checkpoint_path = checkpoint_dir / "dpo_lora.pt"
        temporary_path = checkpoint_dir / "dpo_lora.pt.tmp"
        torch.save(state, temporary_path)
        os.replace(temporary_path, checkpoint_path)
        return checkpoint_dir

    # ------------------------------------------------------------------
    # Шаг 5. Цикл обучения по preference-парам.
    # ------------------------------------------------------------------
    def train(self, records: list[PreferenceRecord]) -> None:
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

        missing_rejected = sum(record.rejected_path is None for record in records)
        if missing_rejected:
            print(
                f"{missing_rejected}/{len(records)} pairs have no rejected image; "
                f"using synthetic {self.config.synthetic_rejected} upscales of x_lr "
                "as y_l (plan: blurred / artifact-prone loser).",
                flush=True,
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
                random.Random(self.config.seed + epoch).shuffle(epoch_records)
                start_index = self.next_record_index if epoch == self.next_epoch else 0

                for record_index in range(start_index, len(epoch_records)):
                    self._raise_if_shutdown_requested()
                    record = epoch_records[record_index]

                    # 5.1. Собираем (x_lr, y_w, y_l).
                    low_resolution = self._prepare_low_resolution_image(record.lr_path)
                    chosen_image = self._prepare_target_image(record.chosen_path)
                    if record.rejected_path is not None:
                        rejected_image = self._prepare_target_image(record.rejected_path)
                    else:
                        rejected_image = self._synthesize_rejected_from_lr(low_resolution)

                    self._raise_if_shutdown_requested()
                    metrics = self._dpo_step(
                        chosen_image,
                        rejected_image,
                        low_resolution,
                        record.prompt,
                    )

                    self.global_step += 1
                    next_record_index = record_index + 1
                    if next_record_index == len(epoch_records):
                        self.next_epoch = epoch + 1
                        self.next_record_index = 0
                    else:
                        self.next_epoch = epoch
                        self.next_record_index = next_record_index

                    print(
                        f"epoch={epoch + 1} step={self.global_step} "
                        f"loss={metrics['loss']:.6f} "
                        f"L_w={metrics['policy_chosen']:.4f} "
                        f"L_l={metrics['policy_rejected']:.4f} "
                        f"margin={metrics['reward_margin']:.4f} "
                        f"acc={metrics['accuracy']:.0f} "
                        f"sigma={metrics['sigma']:.3f}",
                        flush=True,
                    )

                    if self.global_step % self.config.save_every == 0:
                        self._save_checkpoint(
                            f"checkpoint-{self.global_step}", reason="periodic"
                        )
                    self._raise_if_shutdown_requested()

            final_path = self._save_checkpoint("final", reason="completed")
            print(
                f"Flow-DPO training finished. LoRA checkpoint: {final_path}",
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


def parse_args() -> DPOConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default="flow_grpo_dataset/manifest.json",
        help="Preference JSON: lr/prompt/chosen[/rejected], or HR degradation manifest.",
    )
    parser.add_argument(
        "--lr-pipeline",
        help="For an HR-only manifest, train on one degradation pipeline.",
    )
    parser.add_argument("--output-dir", default="dpo_output")
    parser.add_argument(
        "--resume-from-checkpoint",
        help="Full DPO checkpoint directory or dpo_lora.pt file.",
    )
    parser.add_argument(
        "--init-from-lora",
        help=(
            "Load only transformer_lora from a DPO or Flow-GRPO checkpoint "
            "(useful for warm-start / Hybrid chains)."
        ),
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument(
        "--model-revision",
        default=DEFAULT_MODEL_REVISION,
        help="Hugging Face commit/tag for reproducible base-model loading.",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1.0e-5)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument(
        "--beta",
        type=float,
        default=500.0,
        help="DPO temperature β controlling KL stay-near-p_ref strength.",
    )
    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument(
        "--synthetic-rejected",
        choices=["lanczos", "bicubic", "nearest"],
        default="lanczos",
        help="How to build y_l from x_lr when rejected_path is missing.",
    )
    parser.add_argument("--mixed-precision", choices=["fp16", "bf16"], default="fp16")
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    return DPOConfig(
        model_id=args.model_id,
        model_revision=args.model_revision,
        manifest=args.manifest,
        lr_pipeline=args.lr_pipeline,
        output_dir=args.output_dir,
        resume_from_checkpoint=args.resume_from_checkpoint,
        init_from_lora=args.init_from_lora,
        resolution=args.resolution,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        max_grad_norm=args.max_grad_norm,
        beta=args.beta,
        lora_rank=args.lora_rank,
        synthetic_rejected=args.synthetic_rejected,
        mixed_precision=args.mixed_precision,
        save_every=args.save_every,
        max_samples=args.max_samples,
        seed=args.seed,
    )


def main() -> None:
    config = parse_args()
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    records = load_preference_manifest(
        config.manifest, config.max_samples, config.lr_pipeline
    )
    trainer = FlowDPOTrainer(config)
    trainer.train(records)


if __name__ == "__main__":
    main()
