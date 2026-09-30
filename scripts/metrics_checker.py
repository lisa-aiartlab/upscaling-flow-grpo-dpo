from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps
from torch import Tensor


def load_aligned_reference(path: Path, size: tuple[int, int]) -> Tensor:
    """Load an HR image with the same fill-and-center-crop used for LR input."""

    with Image.open(path) as image:
        image = ImageOps.fit(
            image.convert("RGB"),
            size,
            method=Image.Resampling.LANCZOS,
        )
        data = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
        data = data.reshape(image.height, image.width, 3)
    return data.permute(2, 0, 1).float().div(255.0)


class ArtQualityEvaluator:
    """Small, deterministic image-statistics helper used by reward scoring."""

    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device

    @staticmethod
    def compute_edge_sharpness(img_np: np.ndarray) -> float:
        """Return Laplacian variance as a non-directional sharpness measure."""

        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    @staticmethod
    def compute_mean_saturation(img_np: np.ndarray) -> float:
        """Return mean HSV saturation on OpenCV's 0..255 scale."""

        hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
        return float(hsv[:, :, 1].mean())

    @staticmethod
    def _to_uint8_rgb(image: Tensor) -> np.ndarray:
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
    def score_batch_against_reference(
        self,
        images: Tensor,
        reference: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return quality score and HR-relative sharpness/saturation errors."""

        if reference is None:
            zeros = torch.zeros(images.shape[0], dtype=torch.float32)
            return zeros, zeros.clone(), zeros.clone()

        reference_rgb = self._to_uint8_rgb(reference[0])
        target_sharpness = self.compute_edge_sharpness(reference_rgb)
        target_saturation = self.compute_mean_saturation(reference_rgb)
        quality_scores: list[float] = []
        sharpness_errors: list[float] = []
        saturation_errors: list[float] = []

        for image in images:
            generated = self._to_uint8_rgb(image)
            generated_sharpness = self.compute_edge_sharpness(generated)
            generated_saturation = self.compute_mean_saturation(generated)

            # Log space prevents a few high-frequency pixels from dominating
            # Laplacian variance. Saturation already has a bounded 0..255 scale.
            sharpness_error = abs(
                math.log1p(max(0.0, generated_sharpness))
                - math.log1p(max(0.0, target_sharpness))
            ) / 10.0
            saturation_error = abs(
                generated_saturation - target_saturation
            ) / 255.0
            sharpness_errors.append(sharpness_error)
            saturation_errors.append(saturation_error)
            quality_scores.append(-(sharpness_error + saturation_error))

        return (
            torch.tensor(quality_scores, dtype=torch.float32),
            torch.tensor(sharpness_errors, dtype=torch.float32),
            torch.tensor(saturation_errors, dtype=torch.float32),
        )

    def evaluate_pair(
        self, reference_path: str, generated_path: str
    ) -> dict[str, float]:
        """Report comparable generated/reference sharpness and saturation."""

        generated_file = Path(generated_path)
        if not generated_file.is_file():
            raise FileNotFoundError(generated_file)
        with Image.open(generated_file) as image:
            generated = np.asarray(image.convert("RGB")).copy()
        reference_tensor = load_aligned_reference(
            Path(reference_path),
            (generated.shape[1], generated.shape[0]),
        )
        reference = self._to_uint8_rgb(reference_tensor)

        return {
            "sharpness": self.compute_edge_sharpness(generated),
            "reference_sharpness": self.compute_edge_sharpness(reference),
            "saturation": self.compute_mean_saturation(generated),
            "reference_saturation": self.compute_mean_saturation(reference),
        }
