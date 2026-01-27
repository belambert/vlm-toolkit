from io import BytesIO

import torch
import typer
from diffusers import DiffusionPipeline

app = typer.Typer()

from imggen.util import get_device


@app.command()
def main(
    prompt: str = typer.Argument(..., help="The prompt for image generation"),
    output: typer.FileBinaryWrite = typer.Argument(
        "output.png", help="Output filename (use '-' for stdout)"
    ),
):
    pipe = DiffusionPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        # "Tongyi-MAI/Z-Image-Turbo",
        torch_dtype=torch.bfloat16,
    )
    pipe.to(get_device())

    image = pipe(prompt).images[0]

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    output.write(buffer.getvalue())


if __name__ == "__main__":
    app()
