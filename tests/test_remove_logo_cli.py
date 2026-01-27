"""Tests for remove_logo CLI."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from PIL import Image
from typer.testing import CliRunner

from imgproc.cmd.remove_logo import app

runner = CliRunner()


@pytest.fixture
def temp_image(tmp_path):
    """Create a temporary test image."""
    image_path = tmp_path / "test_image.png"
    image = Image.new("RGB", (100, 100), color="white")
    image.save(image_path)
    return image_path


@pytest.fixture
def detections_json(tmp_path, temp_image):
    """Create a temporary detections JSON file."""
    json_path = tmp_path / "detections.json"
    detections = [
        {
            "file_name": str(temp_image),
            "output": json.dumps([{"bbox_2d": [100, 100, 200, 200], "label": "logo"}]),
        }
    ]
    with open(json_path, "w") as f:
        for detection in detections:
            f.write(json.dumps(detection) + "\n")
    return json_path


def test_invalid_method(detections_json, tmp_path):
    """Test that invalid method raises an error."""
    output_dir = tmp_path / "output"

    result = runner.invoke(
        app, [str(detections_json), str(output_dir), "--method", "invalid"]
    )

    assert result.exit_code == 2
    assert "Invalid value for '--method'" in result.output


def test_missing_detections_file(tmp_path):
    """Test that missing detections file raises an error."""
    missing_file = tmp_path / "missing.json"
    output_dir = tmp_path / "output"

    result = runner.invoke(app, [str(missing_file), str(output_dir)])

    assert result.exit_code == 1
    assert "does not exist" in result.stdout


def test_gray_method(detections_json, tmp_path, temp_image):
    """Test processing with gray method."""
    output_dir = tmp_path / "output"

    result = runner.invoke(
        app, [str(detections_json), str(output_dir), "--method", "gray"]
    )

    assert result.exit_code == 0
    assert "Loaded 1 detections" in result.stdout
    assert "Found 1 images with logos to remove" in result.stdout
    assert output_dir.exists()
    assert (output_dir / temp_image.name).exists()


def test_opencv_method(detections_json, tmp_path, temp_image):
    """Test processing with opencv method."""
    output_dir = tmp_path / "output"

    result = runner.invoke(
        app, [str(detections_json), str(output_dir), "--method", "opencv"]
    )

    assert result.exit_code == 0
    assert "Loaded 1 detections" in result.stdout
    assert output_dir.exists()
    assert (output_dir / temp_image.name).exists()


def test_filters_detections_without_bboxes(tmp_path, temp_image):
    """Test that detections without bboxes are filtered out."""
    json_path = tmp_path / "detections.json"
    detections = [
        {
            "file_name": str(temp_image),
            "output": "[]",  # Empty bbox array
        },
        {
            "file_name": str(temp_image),
            "output": json.dumps([{"bbox_2d": [100, 100, 200, 200], "label": "logo"}]),
        },
    ]
    with open(json_path, "w") as f:
        for detection in detections:
            f.write(json.dumps(detection) + "\n")

    output_dir = tmp_path / "output"

    result = runner.invoke(app, [str(json_path), str(output_dir), "--method", "gray"])

    assert result.exit_code == 0
    assert "Loaded 2 detections" in result.stdout
    assert "Found 1 images with logos to remove" in result.stdout


def test_creates_output_directory(detections_json, tmp_path):
    """Test that output directory is created if it doesn't exist."""
    output_dir = tmp_path / "new_output_dir"

    assert not output_dir.exists()

    result = runner.invoke(app, [str(detections_json), str(output_dir)])

    assert result.exit_code == 0
    assert output_dir.exists()


def test_handles_processing_error(tmp_path):
    """Test that processing errors are handled gracefully."""
    # Create detections with invalid image path
    json_path = tmp_path / "detections.json"
    detections = [
        {
            "file_name": "/nonexistent/image.png",
            "output": json.dumps([{"bbox_2d": [100, 100, 200, 200], "label": "logo"}]),
        }
    ]
    with open(json_path, "w") as f:
        for detection in detections:
            f.write(json.dumps(detection) + "\n")

    output_dir = tmp_path / "output"

    result = runner.invoke(app, [str(json_path), str(output_dir), "--method", "gray"])

    assert result.exit_code == 0
    assert "Errors encountered:" in result.stdout
    assert "Processed 0/1 images" in result.stdout


def test_multiple_images(tmp_path):
    """Test processing multiple images."""
    # Create two test images
    image1_path = tmp_path / "image1.png"
    image2_path = tmp_path / "image2.png"
    image = Image.new("RGB", (100, 100), color="white")
    image.save(image1_path)
    image.save(image2_path)

    # Create detections for both
    json_path = tmp_path / "detections.json"
    detections = [
        {
            "file_name": str(image1_path),
            "output": json.dumps([{"bbox_2d": [100, 100, 200, 200], "label": "logo"}]),
        },
        {
            "file_name": str(image2_path),
            "output": json.dumps([{"bbox_2d": [100, 100, 200, 200], "label": "logo"}]),
        },
    ]
    with open(json_path, "w") as f:
        for detection in detections:
            f.write(json.dumps(detection) + "\n")

    output_dir = tmp_path / "output"

    result = runner.invoke(app, [str(json_path), str(output_dir), "--method", "gray"])

    assert result.exit_code == 0
    assert "Loaded 2 detections" in result.stdout
    assert "Found 2 images with logos to remove" in result.stdout
    assert "Processed 2/2 images" in result.stdout
    assert (output_dir / "image1.png").exists()
    assert (output_dir / "image2.png").exists()
