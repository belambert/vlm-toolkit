"""Publish vlm-process output to the Hugging Face Hub as an image dataset."""

from pathlib import Path

from datasets import Dataset, Features, Image, Value

from imgproc.results import load_results


def build_dataset(results: list[dict]) -> Dataset:
    """Build a dataset of images and their VLM output, embedding the image bytes."""
    rows = {
        "image": [str(item["abs_path"]) for item in results],
        # strip for runs produced before vlm-process stripped its own output
        "output": [item["output"].strip() for item in results],
        "file_name": [item["file_name"] for item in results],
    }
    features = Features(
        {"image": Image(), "output": Value("string"), "file_name": Value("string")}
    )
    return Dataset.from_dict(rows, features=features)


def upload(
    results_file: Path,
    repo_id: str,
    private: bool = True,
    split: str = "train",
    token: str | None = None,
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
    return f"https://huggingface.co/datasets/{repo_id}"
