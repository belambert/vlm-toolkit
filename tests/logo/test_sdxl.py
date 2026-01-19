"""Tests for SDXLLogoRemover."""

from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
from PIL import Image

from imggen.logo import SDXLLogoRemover


class TestSDXLLogoRemover:
    """Tests for SDXLLogoRemover."""

    @pytest.fixture
    def mock_pipeline(self):
        """Create a mock pipeline for testing."""
        mock_pipe = MagicMock()
        # Mock the pipeline call to return a result with images
        mock_result = MagicMock()
        mock_result.images = [Image.new("RGB", (100, 100), color="white")]
        mock_pipe.return_value = mock_result
        return mock_pipe

    @pytest.fixture
    def remover(self, mock_pipeline):
        """Create a SDXLLogoRemover with mocked pipeline."""
        with patch(
            "imggen.logo.sdxl.AutoPipelineForInpainting.from_pretrained"
        ) as mock_from_pretrained:
            with patch("imggen.logo.sdxl.get_device", return_value="cpu"):
                mock_from_pretrained.return_value = mock_pipeline
                remover = SDXLLogoRemover("mock-model")
                return remover

    def test_returns_pil_image(self, remover):
        """Test that remove() returns a PIL Image."""
        image = Image.new("RGB", (1000, 1000), color="white")
        bboxes = [{"bbox_2d": [100, 100, 200, 200]}]

        result = remover.remove(image, bboxes)

        assert isinstance(result, Image.Image)

    def test_handles_empty_bboxes(self, remover):
        """Test that remove() handles empty bboxes list."""
        image = Image.new("RGB", (1000, 1000), color="white")
        bboxes = []

        result = remover.remove(image, bboxes)

        assert isinstance(result, Image.Image)

    def test_ensures_dimensions_divisible_by_8(self, remover, mock_pipeline):
        """Test that dimensions are adjusted to be divisible by 8."""
        # Create image with dimensions not divisible by 8
        image = Image.new("RGB", (1003, 1005), color="white")
        bboxes = [{"bbox_2d": [100, 100, 200, 200]}]

        result = remover.remove(image, bboxes)

        # Check that pipeline was called with dimensions divisible by 8
        call_kwargs = mock_pipeline.call_args.kwargs
        assert call_kwargs["width"] % 8 == 0
        assert call_kwargs["height"] % 8 == 0
        # Should be 1000x1000 (cropped from 1003x1005)
        assert call_kwargs["width"] == 1000
        assert call_kwargs["height"] == 1000

    def test_calls_pipeline_with_correct_prompts(self, remover, mock_pipeline):
        """Test that pipeline is called with correct prompts."""
        image = Image.new("RGB", (1000, 1000), color="white")
        bboxes = [{"bbox_2d": [100, 100, 200, 200]}]

        remover.remove(image, bboxes)

        # Check pipeline was called with correct prompts
        call_kwargs = mock_pipeline.call_args.kwargs
        assert call_kwargs["prompt"] == "clean background, no logo, no text"
        assert call_kwargs["negative_prompt"] == "logo, text, signature"

    def test_calls_pipeline_with_correct_parameters(self, remover, mock_pipeline):
        """Test that pipeline is called with correct inference parameters."""
        image = Image.new("RGB", (1000, 1000), color="white")
        bboxes = [{"bbox_2d": [100, 100, 200, 200]}]

        remover.remove(image, bboxes)

        # Check pipeline parameters
        call_kwargs = mock_pipeline.call_args.kwargs
        assert call_kwargs["num_inference_steps"] == 20
        assert call_kwargs["guidance_scale"] == 7.5

    def test_handles_multiple_bboxes(self, remover):
        """Test that remove() handles multiple bboxes."""
        image = Image.new("RGB", (1000, 1000), color="white")
        bboxes = [
            {"bbox_2d": [100, 100, 200, 200]},
            {"bbox_2d": [700, 700, 800, 800]},
        ]

        result = remover.remove(image, bboxes)

        assert isinstance(result, Image.Image)

    def test_pipeline_receives_mask(self, remover, mock_pipeline):
        """Test that pipeline receives a mask image."""
        image = Image.new("RGB", (1000, 1000), color="white")
        bboxes = [{"bbox_2d": [100, 100, 200, 200]}]

        remover.remove(image, bboxes)

        # Check that mask_image was passed
        call_kwargs = mock_pipeline.call_args.kwargs
        assert "mask_image" in call_kwargs
        mask = call_kwargs["mask_image"]
        assert isinstance(mask, Image.Image)
