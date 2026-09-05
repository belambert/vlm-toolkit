import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from PIL import Image


@pytest.fixture
def tmp_images(tmp_path):
    """Create temporary test images and return their paths."""
    paths = []
    for i in range(5):
        p = tmp_path / f"img_{i}.png"
        Image.new("RGB", (100, 100)).save(p)
        paths.append(p)
    return paths


@pytest.fixture
def mock_model():
    model = MagicMock()
    model.device = "cpu"
    # generate returns tensor with shape matching input
    model.generate.return_value = torch.tensor([[1, 2, 3, 10, 20]])
    return model


@pytest.fixture
def mock_processor():
    proc = MagicMock()
    proc.apply_chat_template.return_value = "formatted"
    proc.tokenizer.eos_token_id = 0
    proc.tokenizer.padding_side = "left"
    proc.batch_decode.return_value = ["response text"]

    batch = MagicMock()
    batch.to.return_value = batch
    batch.input_ids = torch.tensor([[1, 2, 3]])
    proc.return_value = batch
    return proc


# --- process_batch tests ---


@patch("qwen_vl_utils.process_vision_info", return_value=(["img"], None))
def test_process_batch_returns_results(mock_vision, mock_model, mock_processor, tmp_images):
    from imgproc.vlm_process import process_batch

    output_file = tmp_images[0].parent / "output.jsonl"
    results = process_batch(
        mock_model, mock_processor, tmp_images[:1], "describe", "cpu", None, output_file
    )
    assert len(results) == 1
    assert results[0]["output"] == "response text"
    assert "file_name" in results[0]


@patch("qwen_vl_utils.process_vision_info", return_value=(["img"], None))
def test_process_batch_strips_whitespace(mock_vision, mock_model, mock_processor, tmp_images):
    from imgproc.vlm_process import process_batch

    mock_processor.batch_decode.return_value = ["  nature\n"]
    output_file = tmp_images[0].parent / "output.jsonl"
    results = process_batch(
        mock_model, mock_processor, tmp_images[:1], "describe", "cpu", None, output_file
    )
    assert results[0]["output"] == "nature"


@patch("qwen_vl_utils.process_vision_info", return_value=(["img"], None))
def test_process_batch_relative_paths(mock_vision, mock_model, mock_processor, tmp_images):
    from imgproc.vlm_process import process_batch

    output_file = tmp_images[0].parent / "output.jsonl"
    results = process_batch(
        mock_model, mock_processor, tmp_images[:1], "describe", "cpu", None, output_file
    )
    # path should be relative to output dir
    rel = results[0]["file_name"]
    assert not Path(rel).is_absolute()


@patch("qwen_vl_utils.process_vision_info", return_value=(["img"], None))
def test_process_batch_all_corrupted(mock_vision, mock_model, mock_processor, tmp_path):
    from imgproc.vlm_process import process_batch

    bad = tmp_path / "bad.png"
    bad.write_bytes(b"garbage")
    output_file = tmp_path / "output.jsonl"

    results = process_batch(
        mock_model, mock_processor, [bad], "describe", "cpu", None, output_file
    )
    assert results == []
    mock_model.generate.assert_not_called()


# --- vlm_process resumption tests ---


@patch("imgproc.vlm_process.load_model")
@patch("imgproc.vlm_process.get_device", return_value="cpu")
@patch("imgproc.vlm_process._run_inference", return_value=[])
@patch("imgproc.vlm_process.prepare_vlm_batch", return_value=(MagicMock(), []))
def test_vlm_process_resumes(mock_prep, mock_infer, mock_device, mock_load, tmp_images):
    from imgproc.vlm_process import vlm_process

    mock_load.return_value = (MagicMock(device="cpu"), MagicMock())

    output = tmp_images[0].parent / "output.jsonl"
    # pre-populate output with 2 processed images
    with open(output, "w") as f:
        for img in tmp_images[:2]:
            rel = img.relative_to(output.parent)
            f.write(json.dumps({"file_name": str(rel), "output": "done"}) + "\n")

    vlm_process(tmp_images[0].parent, output=output, batch_size=1)

    # should only prepare the remaining 3
    total_images = sum(len(call.args[1]) for call in mock_prep.call_args_list)
    assert total_images == 3


@patch("imgproc.vlm_process.load_model")
@patch("imgproc.vlm_process.get_device", return_value="cpu")
def test_vlm_process_skips_when_all_done(mock_device, mock_load, tmp_images):
    from imgproc.vlm_process import vlm_process

    output = tmp_images[0].parent / "output.jsonl"
    with open(output, "w") as f:
        for img in tmp_images:
            rel = img.relative_to(output.parent)
            f.write(json.dumps({"file_name": str(rel), "output": "done"}) + "\n")

    vlm_process(tmp_images[0].parent, output=output)
    # model should never be loaded if all images are done
    mock_load.assert_not_called()


@patch("imgproc.vlm_process.load_model")
@patch("imgproc.vlm_process.get_device", return_value="cpu")
@patch("imgproc.vlm_process._run_inference", return_value=[{"file_name": "x.png", "output": "y"}])
@patch("imgproc.vlm_process.prepare_vlm_batch", return_value=(MagicMock(), ["x.png"]))
def test_vlm_process_batching(mock_prep, mock_infer, mock_device, mock_load, tmp_images):
    from imgproc.vlm_process import vlm_process

    mock_load.return_value = (MagicMock(device="cpu"), MagicMock())

    vlm_process(tmp_images[0].parent, batch_size=2)

    # 5 images with batch_size=2 -> 3 batches (2+2+1)
    assert mock_prep.call_count == 3
    batch_sizes = [len(call.args[1]) for call in mock_prep.call_args_list]
    assert batch_sizes == [2, 2, 1]


@patch("imgproc.vlm_process.load_model")
@patch("imgproc.vlm_process.get_device", return_value="cpu")
@patch("imgproc.vlm_process._run_inference", return_value=[{"file_name": "img.png", "output": "caption"}])
@patch("imgproc.vlm_process.prepare_vlm_batch", return_value=(MagicMock(), ["img.png"]))
def test_vlm_process_writes_jsonl(mock_prep, mock_infer, mock_device, mock_load, tmp_images):
    from imgproc.vlm_process import vlm_process

    mock_load.return_value = (MagicMock(device="cpu"), MagicMock())

    output = tmp_images[0].parent / "output.jsonl"
    vlm_process(tmp_images[0].parent, output=output, batch_size=5)

    lines = output.read_text().strip().split("\n")
    assert len(lines) == 1
    result = json.loads(lines[0])
    assert result["output"] == "caption"
