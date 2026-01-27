import os
from pathlib import Path
from typing import Optional

import torch
import typer
from diffusers import StableDiffusion3Pipeline
from typing_extensions import Annotated

from imggen.util import get_device

# See here for a variety of optimizations:
# https://huggingface.co/docs/diffusers/en/api/pipelines/stable_diffusion/stable_diffusion_3


def cli(
    prompt: Annotated[
        Optional[str], typer.Option(help="a prompt to generate for")
    ] = None,
    output_folder: Annotated[
        Path, typer.Option(exists=False, resolve_path=True)
    ] = Path("."),
    iterations: Annotated[int, typer.Option()] = 30,
    img_per_prompt: Annotated[int, typer.Option()] = 2,
) -> None:
    pipe = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3-medium-diffusers",
        # torch_dtype=torch.float16,
        # text_encoder_3=None,
        # tokenizer_3=None,
        token="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",
    )
    # pipe.enable_model_cpu_offload()

    device = get_device()
    print(device)
    pipe = pipe.to(device)

    os.makedirs(output_folder, exist_ok=True)
    print("generating...")

    images = pipe(
        prompt=prompt,
        negative_prompt="",
        num_inference_steps=iterations,
        height=1024,
        width=1024,
        guidance_scale=7.0,
        num_images_per_prompt=img_per_prompt,
    ).images

    print("saving images...")
    for i, image in enumerate(images):
        image.save(output_folder / f"sd3_{i}.png")


if __name__ == "__main__":
    typer.run(cli)
