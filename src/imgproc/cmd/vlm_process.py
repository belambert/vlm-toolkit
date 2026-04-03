from pathlib import Path

import typer

from imgproc.vlm_process import (
    DEFAULT_MODEL,
    DEFAULT_PROMPT,
    SUGGESTED_MODELS,
    vlm_process,
)

app = typer.Typer()


@app.command()
def main(
    folder: Path = typer.Argument(..., help="Folder containing images to process"),
    output: Path = typer.Option(None, help="Output JSON file"),
    prompt: str = typer.Option(DEFAULT_PROMPT, help="Prompt for the VLM"),
    prompt_file: Path = typer.Option(
        None, help="File containing prompt (overrides --prompt)"
    ),
    model: str = typer.Option(DEFAULT_MODEL, help="HF model"),
    batch_size: int = typer.Option(8, help="Number of images to process in parallel"),
    max_dim: int = typer.Option(
        None, help="Maximum dimension for image resizing (default: 1024)"
    ),
    schema: Path = typer.Option(
        None, help="JSON schema file for constrained decoding (uses outlines)"
    ),
):
    """Process images using a Vision Language Model."""
    if prompt_file is not None:
        prompt = prompt_file.read_text().strip()

    schema_str = schema.read_text() if schema is not None else None

    vlm_process(
        folder=folder,
        output=output,
        prompt=prompt,
        model=model,
        batch_size=batch_size,
        max_dim=max_dim,
        schema=schema_str,
    )


main.__doc__ = f"Process images using a Vision Language Model.\n\n{SUGGESTED_MODELS}"


if __name__ == "__main__":
    app()
