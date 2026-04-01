from pathlib import Path

import typer

from imgproc.vlm_process import vlm_process

app = typer.Typer()

# Get the prompt file path relative to the repository root
PROMPT_FILE = Path(__file__).parent.parent.parent.parent / "prompts" / "caption_img.txt"


@app.command()
def main(
    folder: Path = typer.Argument(..., help="Folder containing images to caption"),
    output: Path = typer.Option(None, help="Output JSON file"),
    model: str = typer.Option(
        "Qwen/Qwen3-VL-2B-Instruct", help="Hugging Face model name"
    ),
    batch_size: int = typer.Option(1, help="Number of images to process in parallel"),
):
    """Generate captions for all images in a folder using a VLM.

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


if __name__ == "__main__":
    app()
