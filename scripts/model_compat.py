"""Compatibility helpers for model-library return types."""

from __future__ import annotations

from typing import Any

from torch import Tensor


def extract_projected_features(output: Tensor | Any) -> Tensor:
    """Return projected features from Transformers 4.x or 5.x CLIP APIs."""

    if isinstance(output, Tensor):
        return output

    pooled_output = getattr(output, "pooler_output", None)
    if isinstance(pooled_output, Tensor):
        return pooled_output

    raise TypeError(
        "CLIP feature output must be a tensor or expose a tensor pooler_output; "
        f"got {type(output).__name__}"
    )
