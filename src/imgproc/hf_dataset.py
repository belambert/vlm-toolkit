"""Publish vlm-process output to the Hugging Face Hub as an image dataset."""

from pathlib import Path

from datasets import Dataset, Features, Image, Value
from huggingface_hub import DatasetCard

from imgproc.results import load_results


def build_dataset(results: list[dict]) -> Dataset:
    """Build a dataset of images and their VLM output, embedding the image bytes."""
    rows = {
        "image": [str(item["abs_path"]) for item in results],
        # strip for runs produced before vlm-process stripped its own output
        "output": [item["output"].strip() for item in results],
        "file_name": [item["abs_path"].name for item in results],
    }
    features = Features(
        {"image": Image(), "output": Value("string"), "file_name": Value("string")}
    )
    return Dataset.from_dict(rows, features=features)


def push_card(repo_id: str, card_file: Path, token: str | None = None) -> None:
    """Attach a dataset card, keeping the dataset_info push_to_hub generated."""
    card = DatasetCard.load(repo_id, token=token)
    incoming = DatasetCard(card_file.read_text())

    # a card file may carry its own frontmatter; layer it over the generated keys
    for key, value in incoming.data.to_dict().items():
        setattr(card.data, key, value)
    card.text = incoming.text

    card.push_to_hub(repo_id, token=token)


def upload(
    results_file: Path,
    repo_id: str,
    private: bool = True,
    split: str = "train",
    token: str | None = None,
    card_file: Path | None = None,
) -> str:
    """Upload a vlm-process JSONL run to repo_id, returning the dataset URL."""
    results = load_results(results_file)
    if not results:
        raise ValueError(f"{results_file} contains no results")

    missing = [item for item in results if not item["abs_path"].is_file()]
    if missing:
        for item in missing[:10]:
            print(f"Skipping missing image: {item['file_name']}", flush=True)
        if len(missing) > 10:
            print(f"...and {len(missing) - 10} more", flush=True)
        results = [item for item in results if item["abs_path"].is_file()]

    if not results:
        raise ValueError("none of the referenced images exist on disk")

    print(f"Uploading {len(results)} images to {repo_id}", flush=True)
    ds = build_dataset(results)
    # embed_external_files inlines the image bytes; without it only paths ship
    ds.push_to_hub(
        repo_id,
        private=private,
        split=split,
        token=token,
        embed_external_files=True,
    )

    if card_file is not None:
        push_card(repo_id, card_file, token=token)
        print(f"Attached dataset card from {card_file}", flush=True)

    return f"https://huggingface.co/datasets/{repo_id}"
