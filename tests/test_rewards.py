from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image, ImageOps


PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
SCRIPTS_DIRECTORY = PROJECT_DIRECTORY / "scripts"
sys.path.insert(0, str(SCRIPTS_DIRECTORY))

from metrics_checker import ArtQualityEvaluator, load_aligned_reference
from model_compat import extract_projected_features
from training_scripts.flow_grpo import UpscalingReward


class UpscalingRewardTests(unittest.TestCase):
    def setUp(self) -> None:
        self.evaluator = ArtQualityEvaluator(device="cpu")

    def test_reference_uses_same_center_crop_as_lr_preprocessing(self) -> None:
        source = np.zeros((20, 40, 3), dtype=np.uint8)
        source[:, :10] = (255, 0, 0)
        source[:, 10:30] = (0, 255, 0)
        source[:, 30:] = (0, 0, 255)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reference.png"
            Image.fromarray(source).save(path)
            actual = load_aligned_reference(path, (16, 16))

        expected_image = ImageOps.fit(
            Image.fromarray(source),
            (16, 16),
            method=Image.Resampling.LANCZOS,
        )
        expected = torch.from_numpy(np.asarray(expected_image).copy())
        expected = expected.permute(2, 0, 1).float().div(255.0)
        torch.testing.assert_close(actual, expected)

    def test_identical_reference_has_zero_quality_error(self) -> None:
        generator = torch.Generator().manual_seed(7)
        reference = torch.rand((1, 3, 32, 32), generator=generator)
        images = reference.repeat(2, 1, 1, 1)

        scores, sharpness_errors, saturation_errors = (
            self.evaluator.score_batch_against_reference(images, reference)
        )

        torch.testing.assert_close(scores, torch.zeros(2))
        torch.testing.assert_close(sharpness_errors, torch.zeros(2))
        torch.testing.assert_close(saturation_errors, torch.zeros(2))

    def test_missing_reference_disables_target_dependent_metrics(self) -> None:
        images = torch.rand((3, 3, 16, 16))
        scores, sharpness_errors, saturation_errors = (
            self.evaluator.score_batch_against_reference(images, None)
        )

        torch.testing.assert_close(scores, torch.zeros(3))
        torch.testing.assert_close(sharpness_errors, torch.zeros(3))
        torch.testing.assert_close(saturation_errors, torch.zeros(3))

    def test_quality_score_penalizes_distance_from_hr_target(self) -> None:
        reference = torch.zeros((1, 3, 16, 16))
        reference[:, 0] = 1.0
        generated = torch.full((1, 3, 16, 16), 0.5)

        scores, sharpness_errors, saturation_errors = (
            self.evaluator.score_batch_against_reference(generated, reference)
        )

        self.assertLess(float(scores[0]), 0.0)
        self.assertEqual(float(sharpness_errors[0]), 0.0)
        self.assertGreater(float(saturation_errors[0]), 0.9)

    def test_clip_feature_compatibility_accepts_transformers_4_tensor(self) -> None:
        expected = torch.rand((2, 4))

        actual = extract_projected_features(expected)

        self.assertIs(actual, expected)

    def test_clip_feature_compatibility_accepts_transformers_5_output(self) -> None:
        expected = torch.rand((2, 4))
        output = SimpleNamespace(pooler_output=expected)

        actual = extract_projected_features(output)

        self.assertIs(actual, expected)

    def test_clip_reward_truncates_prompts_to_model_limit(self) -> None:
        class RecordingProcessor:
            def __init__(self) -> None:
                self.kwargs: dict[str, object] = {}

            def __call__(self, **kwargs: object) -> dict[str, torch.Tensor]:
                self.kwargs = kwargs
                sequence_length = int(kwargs["max_length"])
                return {
                    "pixel_values": torch.zeros((1, 3, 16, 16)),
                    "input_ids": torch.zeros((1, sequence_length), dtype=torch.long),
                    "attention_mask": torch.ones((1, sequence_length), dtype=torch.long),
                }

        class FakeClipModel:
            config = SimpleNamespace(
                text_config=SimpleNamespace(max_position_embeddings=77)
            )

            @staticmethod
            def get_image_features(**_: torch.Tensor) -> torch.Tensor:
                return torch.ones((1, 4))

            @staticmethod
            def get_text_features(**kwargs: torch.Tensor) -> torch.Tensor:
                if kwargs["input_ids"].shape[1] > 77:
                    raise ValueError("prompt exceeded the CLIP context window")
                return torch.ones((1, 4))

        reward = UpscalingReward.__new__(UpscalingReward)
        reward.device = torch.device("cpu")
        reward.clip_processor = RecordingProcessor()
        reward.clip_model = FakeClipModel()

        scores = reward._prompt_scores(
            torch.zeros((1, 3, 16, 16)), "long prompt " * 100
        )

        self.assertEqual(reward.clip_processor.kwargs["max_length"], 77)
        self.assertIs(reward.clip_processor.kwargs["truncation"], True)
        torch.testing.assert_close(scores, torch.ones(1))


if __name__ == "__main__":
    unittest.main()
