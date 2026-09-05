from importlib.resources import files
from pathlib import Path

import typer

from imgproc.vlm_process import DEFAULT_MODEL, SUGGESTED_MODELS, vlm_process

app = typer.Typer()

PROMPT_FILE = files("imgproc") / "prompts" / "detect_logo.txt"


@app.command()
def main(
    folder: Path = typer.Argument(..., help="Folder containing images to check"),
    output: Path = typer.Option(None, help="Output JSON file"),
    model: str = typer.Option(DEFAULT_MODEL, help="HF model"),
    batch_size: int = typer.Option(8, help="Number of images to process in parallel"),
):
    """Detect logos with a VLM."""
    # Read prompt from file
    prompt = PROMPT_FILE.read_text()

    # Set default output filename for detect-logo
    if output is None:
        output = folder / "logo_bbox_output.json"

    vlm_process(
        folder=folder,
        output=output,
        prompt=prompt,
        model=model,
        batch_size=batch_size,
    )


main.__doc__ = f"Detect logos with a VLM.\n\n{SUGGESTED_MODELS}"


if __name__ == "__main__":
    app()
