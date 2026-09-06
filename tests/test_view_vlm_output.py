import json

import pytest
from PIL import Image

from vlm_toolkit.cmd.view_vlm_output import _render, _web_root
from vlm_toolkit.results import load_results


@pytest.fixture
def outputs(tmp_path):
    """A JSONL file whose file_name entries escape the JSON's own directory."""
    (tmp_path / "out").mkdir()
    (tmp_path / "imgs").mkdir()
    for n in ("a", "b"):
        Image.new("RGB", (10, 10)).save(tmp_path / "imgs" / f"{n}.jpg")

    jsonl = tmp_path / "out" / "o.jsonl"
    with open(jsonl, "w") as f:
        for n in ("a", "b"):
            f.write(
                json.dumps({"file_name": f"../imgs/{n}.jpg", "output": f"cap {n}"})
                + "\n"
            )
    return jsonl


def test_load_items_resolves_relative_paths(outputs):
    items = load_results(outputs)
    assert len(items) == 2
    assert items[0]["abs_path"].is_file()
    assert items[0]["abs_path"].name == "a.jpg"


def test_web_root_covers_images_outside_json_dir(outputs):
    items = load_results(outputs)
    root, srcs = _web_root(items, outputs.parent)

    # root must contain every image, so it cannot be the json dir itself
    assert root != outputs.parent
    for item in items:
        assert item["abs_path"].is_relative_to(root)
    assert srcs == ["imgs/a.jpg", "imgs/b.jpg"]


def test_web_root_quotes_urls(tmp_path):
    (tmp_path / "my imgs").mkdir()
    img = tmp_path / "my imgs" / "a b.jpg"
    Image.new("RGB", (10, 10)).save(img)
    items = [{"abs_path": img, "output": "x"}]

    _, srcs = _web_root(items, tmp_path)
    assert srcs == ["my%20imgs/a%20b.jpg"]


def test_render_uses_given_srcs(outputs):
    items = load_results(outputs)
    html = _render(items, ["imgs/a.jpg", "imgs/b.jpg"])
    assert '<img src="imgs/a.jpg"' in html
    assert "cap a" in html
    assert "cap b" in html


def test_render_escapes_model_output(tmp_path):
    items = [{"abs_path": tmp_path / "a.jpg", "output": "<script>alert(1)</script>"}]
    html = _render(items, ["a.jpg"])
    assert "<script>" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
