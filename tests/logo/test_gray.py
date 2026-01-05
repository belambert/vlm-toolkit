"""Tests for GrayLogoRemover."""

import numpy as np
import pytest
from PIL import Image

from imggen.logo import GrayLogoRemover


class TestGrayLogoRemover:
    """Tests for GrayLogoRemover."""

    def test_returns_pil_image(self):
        """Test that remove() returns a PIL Image."""
        remover = GrayLogoRemover()
        image = Image.new("RGB", (100, 100), color="white")
        bboxes = [{"bbox_2d": [100, 100, 200, 200]}]

        result = remover.remove(image, bboxes)

        assert isinstance(result, Image.Image)

    def test_handles_empty_bboxes(self):
        """Test that remove() handles empty bboxes list."""
        remover = GrayLogoRemover()
        image = Image.new("RGB", (100, 100), color="white")
        bboxes = []

        result = remover.remove(image, bboxes)

        assert isinstance(result, Image.Image)
        # Image should be unchanged
        assert np.array_equal(np.array(result), np.array(image))

    def test_draws_gray_rectangle(self):
        """Test that remove() draws gray rectangle over bbox."""
        remover = GrayLogoRemover()
        # Create a white image
        image = Image.new("RGB", (1000, 1000), color=(255, 255, 255))
        # Bbox covering center 200x200 pixels (in 1000x1000 normalized coords)
        bboxes = [{"bbox_2d": [400, 400, 600, 600]}]

        result = remover.remove(image, bboxes)

        # Convert to numpy array for pixel checking
        result_array = np.array(result)

        # Check that center area has gray pixels (accounting for 15% buffer)
        # Center should definitely be gray
        center_pixel = result_array[500, 500]
        assert center_pixel[0] == 128
        assert center_pixel[1] == 128
        assert center_pixel[2] == 128

    def test_handles_multiple_bboxes(self):
        """Test that remove() handles multiple bboxes."""
        remover = GrayLogoRemover()
        image = Image.new("RGB", (1000, 1000), color="white")
        bboxes = [
            {"bbox_2d": [100, 100, 200, 200]},
            {"bbox_2d": [700, 700, 800, 800]},
        ]

        result = remover.remove(image, bboxes)

        assert isinstance(result, Image.Image)
        result_array = np.array(result)

        # Check both bbox areas have gray pixels
        # First bbox center (accounting for scaling and buffer)
        pixel1 = result_array[150, 150]
        assert pixel1[0] == 128

        # Second bbox center
        pixel2 = result_array[750, 750]
        assert pixel2[0] == 128

    def test_ignores_bbox_without_bbox_2d(self):
        """Test that bboxes without bbox_2d field are ignored."""
        remover = GrayLogoRemover()
        image = Image.new("RGB", (100, 100), color="white")
        bboxes = [{"label": "logo"}]  # Missing bbox_2d

        result = remover.remove(image, bboxes)

        assert isinstance(result, Image.Image)
        # Image should be unchanged
        assert np.array_equal(np.array(result), np.array(image))
