from importlib.resources import files
from pathlib import Path

import typer

from imgproc.vlm_process import DEFAULT_MODEL, SUGGESTED_MODELS, vlm_process

app = typer.Typer()

PROMPT_FILE = files("imgproc") / "prompts" / "caption.txt"


@app.command()
def main(
    folder: Path = typer.Argument(..., help="Folder containing images to caption"),
    output: Path = typer.Option(None, help="Output JSON file"),
    model: str = typer.Option(DEFAULT_MODEL, help="Hugging Face model name"),
    batch_size: int = typer.Option(1, help="Number of images to process in parallel"),
) -> None:
    """Generate captions for all images in a folder using a VLM."""
    prompt = PROMPT_FILE.read_text()

    if output is None:
        output = folder / "captions.json"

    vlm_process(
        folder=folder,
        output=output,
        prompt=prompt,
        model=model,
        batch_size=batch_size,
    )


main.__doc__ = (
    f"Generate captions for all images in a folder using a VLM.\n\n{SUGGESTED_MODELS}"
)


if __name__ == "__main__":
    app()
