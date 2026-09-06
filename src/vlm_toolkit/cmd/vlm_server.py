from pathlib import Path

import typer

from vlm_toolkit.vlm_server import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    DEFAULT_PROMPT,
    vlm_server_process,
)

app = typer.Typer()


@app.command()
def main(
    folder: Path = typer.Argument(..., help="Folder containing images to process"),
    output: Path = typer.Option(None, help="Output JSONL file"),
    prompt: str = typer.Option(DEFAULT_PROMPT, help="Prompt for the VLM"),
    prompt_file: Path = typer.Option(
        None, help="File containing prompt (overrides --prompt)"
    ),
    base_url: str = typer.Option(DEFAULT_BASE_URL, help="Server base URL"),
    model: str = typer.Option(DEFAULT_MODEL, help="Model name to send in requests"),
    max_dim: int = typer.Option(
        None, help="Maximum dimension for image resizing (default: 1024)"
    ),
    max_tokens: int = typer.Option(512, help="Max tokens to generate"),
    concurrency: int = typer.Option(8, help="Number of concurrent requests"),
) -> None:
    """Process images via an OpenAI-compatible vision endpoint."""
    if prompt_file is not None:
        prompt = prompt_file.read_text().strip()

    vlm_server_process(
        folder=folder,
        output=output,
        prompt=prompt,
        base_url=base_url,
        model=model,
        max_dim=max_dim,
        max_tokens=max_tokens,
        concurrency=concurrency,
    )


if __name__ == "__main__":
    app()
