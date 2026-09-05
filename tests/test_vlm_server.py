from unittest.mock import MagicMock

import pytest
from PIL import Image

from imgproc.vlm_server import _process_image


@pytest.fixture
def tmp_image(tmp_path):
    p = tmp_path / "img.png"
    Image.new("RGB", (100, 100)).save(p)
    return p


def _client(content):
    resp = MagicMock()
    resp.json.return_value = {"choices": [{"message": {"content": content}}]}
    client = MagicMock()
    client.post.return_value = resp
    return client


def test_process_image_strips_whitespace(tmp_image):
    client = _client("  nature\n")
    result = _process_image(
        client, "http://x/v1", "m", "describe", tmp_image, None, 512
    )
    assert result == "nature"


def test_process_image_returns_none_for_corrupt_image(tmp_path):
    bad = tmp_path / "bad.png"
    bad.write_text("not an image")
    result = _process_image(_client("x"), "http://x/v1", "m", "p", bad, None, 512)
    assert result is None
