from pathlib import Path

import typer

app = typer.Typer()


@app.command()
def main(
    results_file: Path = typer.Argument(..., help="JSONL output from vlm-process"),
    repo_id: str = typer.Argument(
        ..., help="Target dataset repo, e.g. user/my-dataset"
    ),
    private: bool = typer.Option(True, help="Create the dataset repo as private"),
    split: str = typer.Option("train", help="Split name to upload as"),
    token: str = typer.Option(
        None, help="Hugging Face token (defaults to HF_TOKEN or a cached login)"
    ),
    card: Path = typer.Option(
        None, help="Markdown file to attach as the dataset card (README.md)"
    ),
) -> None:
    """Upload a vlm-process run to the Hugging Face Hub as an image dataset.

    Images are embedded in the dataset, not referenced by path, so the result is
    self-contained and works in the Hub's dataset viewer. --card attaches a
    README, preserving the dataset_info metadata the upload generates.
    """
    # deferred so `vlm --help` works without the hub extra installed
    from vlm_tools.hf_dataset import upload

    if not results_file.exists():
        print(f"Error: {results_file} does not exist", flush=True)
        raise typer.Exit(1)

    # check before uploading, so a typo doesn't surface after a long transfer
    if card is not None and not card.exists():
        print(f"Error: {card} does not exist", flush=True)
        raise typer.Exit(1)

    url = upload(
        results_file,
        repo_id,
        private=private,
        split=split,
        token=token,
        card_file=card,
    )
    print(f"Uploaded to {url}", flush=True)


if __name__ == "__main__":
    app()
