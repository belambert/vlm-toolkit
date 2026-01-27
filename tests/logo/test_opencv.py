"""Tests for OpenCVLogoRemover."""

import numpy as np
import pytest
from PIL import Image

from imgproc.logo import OpenCVLogoRemover


class TestOpenCVLogoRemover:
    """Tests for OpenCVLogoRemover."""

    def test_returns_pil_image(self):
        """Test that remove() returns a PIL Image."""
        remover = OpenCVLogoRemover()
        image = Image.new("RGB", (100, 100), color="white")
        bboxes = [{"bbox_2d": [100, 100, 200, 200]}]

        result = remover.remove(image, bboxes)

        assert isinstance(result, Image.Image)

    def test_handles_empty_bboxes(self):
        """Test that remove() handles empty bboxes list."""
        remover = OpenCVLogoRemover()
        image = Image.new("RGB", (100, 100), color="white")
        bboxes = []

        result = remover.remove(image, bboxes)

        assert isinstance(result, Image.Image)
        # Image should be unchanged
        assert np.array_equal(np.array(result), np.array(image))

    def test_inpaints_bbox_area(self):
        """Test that remove() inpaints the bbox area."""
        remover = OpenCVLogoRemover()
        # Create an image with a black square in the center
        image = Image.new("RGB", (1000, 1000), color=(255, 255, 255))
        image_array = np.array(image)
        # Draw a black square in center
        image_array[400:600, 400:600] = [0, 0, 0]
        image = Image.fromarray(image_array)

        # Bbox covering the black square
        bboxes = [{"bbox_2d": [400, 400, 600, 600]}]

        result = remover.remove(image, bboxes)

        assert isinstance(result, Image.Image)
        result_array = np.array(result)

        # After inpainting, the center should no longer be pure black
        # OpenCV inpainting should blend with surrounding white pixels
        center_pixel = result_array[500, 500]
        # Check that it's been modified (not pure black anymore)
        assert not (
            center_pixel[0] == 0 and center_pixel[1] == 0 and center_pixel[2] == 0
        )

    def test_handles_multiple_bboxes(self):
        """Test that remove() handles multiple bboxes."""
        remover = OpenCVLogoRemover()
        image = Image.new("RGB", (1000, 1000), color="white")
        image_array = np.array(image)
        # Draw two black squares
        image_array[100:200, 100:200] = [0, 0, 0]
        image_array[700:800, 700:800] = [0, 0, 0]
        image = Image.fromarray(image_array)

        bboxes = [
            {"bbox_2d": [100, 100, 200, 200]},
            {"bbox_2d": [700, 700, 800, 800]},
        ]

        result = remover.remove(image, bboxes)

        assert isinstance(result, Image.Image)

    def test_ignores_bbox_without_bbox_2d(self):
        """Test that bboxes without bbox_2d field are ignored."""
        remover = OpenCVLogoRemover()
        image = Image.new("RGB", (100, 100), color="white")
        bboxes = [{"label": "logo"}]  # Missing bbox_2d

        result = remover.remove(image, bboxes)

        assert isinstance(result, Image.Image)
        # Image should be unchanged
        assert np.array_equal(np.array(result), np.array(image))

    def test_preserves_image_dimensions(self):
        """Test that output image has same dimensions as input."""
        remover = OpenCVLogoRemover()
        image = Image.new("RGB", (500, 300), color="white")
        bboxes = [{"bbox_2d": [200, 200, 400, 400]}]

        result = remover.remove(image, bboxes)

        assert result.size == image.size
