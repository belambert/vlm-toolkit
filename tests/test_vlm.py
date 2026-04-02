from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from imgproc.vlm import prepare_vlm_batch


@pytest.fixture
def processor():
    proc = MagicMock()
    proc.apply_chat_template.return_value = "formatted text"
    proc.return_value = MagicMock()  # processor() call returns batch inputs
    proc.return_value.to.return_value = proc.return_value
    return proc


@pytest.fixture
def tmp_images(tmp_path):
    """Create temporary test images and return their paths."""
    paths = []
    for i in range(3):
        p = tmp_path / f"img_{i}.png"
        Image.new("RGB", (100, 100), color=(i * 50, 0, 0)).save(p)
        paths.append(p)
    return paths


@patch("qwen_vl_utils.process_vision_info", return_value=(["img_input"], None))
def test_prepare_vlm_batch_loads_all_images(mock_vision, processor, tmp_images):
    inputs, valid_paths = prepare_vlm_batch(
        processor, tmp_images, "describe", "cpu"
    )
    assert len(valid_paths) == 3
    assert valid_paths == tmp_images
    assert processor.apply_chat_template.call_count == 3
    assert processor.call_count == 1


@patch("qwen_vl_utils.process_vision_info", return_value=(["img_input"], None))
def test_prepare_vlm_batch_skips_corrupted(mock_vision, processor, tmp_images):
    # write garbage to the second image
    tmp_images[1].write_bytes(b"not an image")

    inputs, valid_paths = prepare_vlm_batch(
        processor, tmp_images, "describe", "cpu"
    )
    assert len(valid_paths) == 2
    assert tmp_images[1] not in valid_paths


@patch("qwen_vl_utils.process_vision_info", return_value=(["img_input"], None))
def test_prepare_vlm_batch_all_corrupted(mock_vision, processor, tmp_path):
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"garbage")

    inputs, valid_paths = prepare_vlm_batch(
        processor, [bad], "describe", "cpu"
    )
    assert inputs is None
    assert valid_paths == []


@patch("qwen_vl_utils.process_vision_info", return_value=(["img_input"], None))
def test_prepare_vlm_batch_respects_max_dim(mock_vision, processor, tmp_path):
    p = tmp_path / "big.png"
    Image.new("RGB", (2000, 1000)).save(p)

    inputs, valid_paths = prepare_vlm_batch(
        processor, [p], "describe", "cpu", max_dim=512
    )
    # check that the image passed to messages was resized
    call_args = processor.apply_chat_template.call_args
    messages = call_args[0][0]
    img = messages[0]["content"][0]["image"]
    assert img.size[0] <= 512
    assert img.size[1] <= 512


@patch("qwen_vl_utils.process_vision_info", return_value=(["img_input"], None))
def test_prepare_vlm_batch_moves_to_device(mock_vision, processor, tmp_images):
    inputs, _ = prepare_vlm_batch(
        processor, tmp_images[:1], "describe", "cpu"
    )
    processor.return_value.to.assert_called_once_with("cpu")
