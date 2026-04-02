from pathlib import Path

import typer

from imgproc.vlm_process import DEFAULT_MODEL, SUGGESTED_MODELS, vlm_process

app = typer.Typer()

# Get the prompt file path relative to the repository root
PROMPT_FILE = Path(__file__).parent.parent.parent.parent / "prompts" / "caption_img.txt"


@app.command()
def main(
    folder: Path = typer.Argument(..., help="Folder containing images to caption"),
    output: Path = typer.Option(None, help="Output JSON file"),
    model: str = typer.Option(DEFAULT_MODEL, help="Hugging Face model name"),
    batch_size: int = typer.Option(1, help="Number of images to process in parallel"),
):
    """Generate captions for all images in a folder using a VLM."""
    # Read prompt from file
    prompt = PROMPT_FILE.read_text()

    # Set default output filename for caption-images
    if output is None:
        output = folder / "captions.json"

    vlm_process(
        folder=folder,
        output=output,
        prompt=prompt,
        model=model,
        batch_size=batch_size,
    )


main.__doc__ = f"Generate captions for all images in a folder using a VLM.\n\n{SUGGESTED_MODELS}"


if __name__ == "__main__":
    app()
