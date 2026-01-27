from io import BytesIO

import torch
import typer
from diffusers import DiffusionPipeline
from diffusers.utils import load_image

from imggen.util import get_device

app = typer.Typer()


@app.command()
def main(
    # prompt: str = typer.Argument(..., help="The prompt for image generation"),
    output: typer.FileBinaryWrite = typer.Argument(
        "output.png", help="Output filename (use '-' for stdout)"
    ),
):
    device = get_device()
    pipe = DiffusionPipeline.from_pretrained(
        "Qwen/Qwen-Image-Edit",
        dtype=torch.bfloat16,
        # dtype=torch.float8_e4m3fn,
        device_map=device,
    )

    prompt = "Turn this cat into a dog"

    input_image = load_image(
        "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png"
    )

    image = pipe(image=input_image, prompt=prompt).images[0]

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    output.write(buffer.getvalue())


if __name__ == "__main__":
    app()
