import json
from unittest.mock import patch

import pytest
from huggingface_hub import DatasetCard
from PIL import Image

from imgproc.hf_dataset import build_dataset, push_card, upload
from imgproc.results import load_results


@pytest.fixture
def run_dir(tmp_path):
    """A vlm-process run: two images plus a JSONL referencing them via ../."""
    (tmp_path / "out").mkdir()
    (tmp_path / "imgs").mkdir()
    for n in ("a", "b"):
        Image.new("RGB", (16, 12)).save(tmp_path / "imgs" / f"{n}.jpg")

    jsonl = tmp_path / "out" / "o.jsonl"
    with open(jsonl, "w") as f:
        for n in ("a", "b"):
            f.write(
                json.dumps({"file_name": f"../imgs/{n}.jpg", "output": f"cap {n}"})
                + "\n"
            )
    return jsonl


def test_build_dataset_columns_and_decoding(run_dir):
    ds = build_dataset(load_results(run_dir))
    assert ds.num_rows == 2
    assert set(ds.column_names) == {"image", "output", "file_name"}
    assert ds[0]["image"].size == (16, 12)
    assert ds[0]["output"] == "cap a"


def test_build_dataset_uses_basenames(run_dir):
    ds = build_dataset(load_results(run_dir))
    # the source jsonl refers to the images as ../imgs/<name>.jpg
    assert ds["file_name"] == ["a.jpg", "b.jpg"]


def test_build_dataset_strips_legacy_whitespace(tmp_path):
    img = tmp_path / "a.jpg"
    Image.new("RGB", (8, 8)).save(img)
    ds = build_dataset([{"abs_path": img, "output": "nature\n", "file_name": "a.jpg"}])
    assert ds[0]["output"] == "nature"


def test_upload_embeds_images(run_dir):
    with patch("imgproc.hf_dataset.Dataset.push_to_hub") as push:
        url = upload(run_dir, "user/ds", private=True)

    assert url == "https://huggingface.co/datasets/user/ds"
    assert push.call_args.kwargs["embed_external_files"] is True
    assert push.call_args.kwargs["private"] is True


def test_upload_skips_missing_images(run_dir, capsys):
    (run_dir.parent.parent / "imgs" / "a.jpg").unlink()

    with patch("imgproc.hf_dataset.Dataset.push_to_hub"):
        upload(run_dir, "user/ds")

    assert "Skipping missing image" in capsys.readouterr().out


def test_upload_attaches_card_when_given(run_dir, tmp_path):
    card = tmp_path / "CARD.md"
    card.write_text("# Plants\n\nA test dataset.\n")

    with patch("imgproc.hf_dataset.Dataset.push_to_hub"):
        with patch("imgproc.hf_dataset.push_card") as push_card:
            upload(run_dir, "user/ds", card_file=card)

    push_card.assert_called_once()
    assert push_card.call_args.args[1] == card


def test_upload_skips_card_by_default(run_dir):
    with patch("imgproc.hf_dataset.Dataset.push_to_hub"):
        with patch("imgproc.hf_dataset.push_card") as push_card:
            upload(run_dir, "user/ds")

    push_card.assert_not_called()


def test_push_card_preserves_generated_metadata(tmp_path):
    card_file = tmp_path / "CARD.md"
    card_file.write_text("# Title\n\nBody text.\n")

    existing = DatasetCard("---\ndataset_info:\n  splits: []\n---\n")
    with patch("imgproc.hf_dataset.DatasetCard.load", return_value=existing):
        with patch.object(DatasetCard, "push_to_hub") as push:
            push_card("user/ds", card_file)

    assert "Body text." in existing.text
    # the generated dataset_info must survive the card overwrite
    assert existing.data.to_dict()["dataset_info"] == {"splits": []}
    push.assert_called_once()


def test_push_card_merges_frontmatter(tmp_path):
    card_file = tmp_path / "CARD.md"
    card_file.write_text("---\nlicense: mit\n---\n\nBody.\n")

    existing = DatasetCard("---\ndataset_info:\n  splits: []\n---\n")
    with patch("imgproc.hf_dataset.DatasetCard.load", return_value=existing):
        with patch.object(DatasetCard, "push_to_hub"):
            push_card("user/ds", card_file)

    data = existing.data.to_dict()
    assert data["license"] == "mit"
    assert data["dataset_info"] == {"splits": []}


def test_upload_rejects_empty_results(tmp_path):
    empty = tmp_path / "empty.jsonl"
    empty.write_text("")
    with pytest.raises(ValueError, match="no results"):
        upload(empty, "user/ds")


def test_upload_rejects_all_images_missing(run_dir):
    for n in ("a", "b"):
        (run_dir.parent.parent / "imgs" / f"{n}.jpg").unlink()
    with pytest.raises(ValueError, match="none of the referenced images"):
        upload(run_dir, "user/ds")
