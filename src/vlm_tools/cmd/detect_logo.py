from importlib.resources import files
from pathlib import Path

import typer

from vlm_tools.models import DEFAULT_MODEL, SUGGESTED_MODELS

app = typer.Typer()

PROMPT_FILE = files("vlm_tools") / "prompts" / "detect_logo.txt"


@app.command()
def main(
    folder: Path = typer.Argument(..., help="Folder containing images to check"),
    output: Path = typer.Option(None, help="Output JSON file"),
    model: str = typer.Option(DEFAULT_MODEL, help="HF model"),
    batch_size: int = typer.Option(8, help="Number of images to process in parallel"),
) -> None:
    """Detect logos with a VLM."""
    # deferred so `vlm --help` doesn't pay for the torch import
    from vlm_tools.vlm_process import vlm_process

    prompt = PROMPT_FILE.read_text()

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
