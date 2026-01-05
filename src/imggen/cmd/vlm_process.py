from pathlib import Path

import typer

from imggen.vlm_process import DEFAULT_PROMPT, vlm_process

app = typer.Typer()


@app.command()
def main(
    folder: Path = typer.Argument(..., help="Folder containing images to process"),
    output: Path = typer.Option(None, help="Output JSON file"),
    prompt: str = typer.Option(DEFAULT_PROMPT, help="Prompt for the VLM"),
    prompt_file: Path = typer.Option(
        None, help="File containing prompt (overrides --prompt)"
    ),
    model_name: str = typer.Option("Qwen/Qwen3-VL-2B-Instruct", help="HF model"),
    batch_size: int = typer.Option(8, help="Number of images to process in parallel"),
):
    """Process images using a Vision Language Model.

    Some good models to use:

    - Qwen/Qwen3-VL-2B-Instruct
    - Qwen/Qwen3-VL-4B-Instruct
    - Qwen/Qwen3-VL-8B-Instruct
    - Qwen/Qwen3-VL-32B-Instruct

    All of these have quantized versions, which can be specified by adding "-FP8" to
    the end of the name.
    """
    # Read prompt from file if provided
    if prompt_file is not None:
        prompt = prompt_file.read_text().strip()

    vlm_process(
        folder=folder,
        output=output,
        prompt=prompt,
        model_name=model_name,
        batch_size=batch_size,
    )


if __name__ == "__main__":
    app()
