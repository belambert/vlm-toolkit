from pathlib import Path

import typer

from imggen.vlm_process import vlm_process

app = typer.Typer()

# Get the prompt file path relative to the repository root
PROMPT_FILE = Path(__file__).parent.parent.parent.parent / "prompts" / "detect_logo.txt"


@app.command()
def main(
    folder: Path = typer.Argument(..., help="Folder containing images to check"),
    output: Path = typer.Option(None, help="Output JSON file"),
    model_name: str = typer.Option("Qwen/Qwen3-VL-2B-Instruct", help="HF model"),
    batch_size: int = typer.Option(1, help="Number of images to process in parallel"),
):
    """Detect logos with a VLM.

    Some good models to use:

    - Qwen/Qwen3-VL-2B-Instruct
    - Qwen/Qwen3-VL-4B-Instruct
    - Qwen/Qwen3-VL-8B-Instruct
    - Qwen/Qwen3-VL-32B-Instruct

    All of these have quantized versions, which can be specified by adding "-FP8" to
    the end of the name.
    """
    # Read prompt from file
    prompt = PROMPT_FILE.read_text()

    # Set default output filename for detect-logo
    if output is None:
        output = folder / "logo_detections.json"

    vlm_process(
        folder=folder,
        output=output,
        prompt=prompt,
        model_name=model_name,
        batch_size=batch_size,
    )


if __name__ == "__main__":
    app()
