import os
import re
from base64 import b64encode
from pathlib import Path
from typing import Optional

import torch
import typer
from diffusers import (  # StableDiffusionXLPipeline,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
)
from typing_extensions import Annotated

from imggen.util import get_device

# pipeline = StableDiffusionPipeline.from_single_file(
#     "https://huggingface.co/WarriorMama777/OrangeMixs/blob/main/Models/"
# "AbyssOrangeMix/AbyssOrangeMix.safetensors"
# )


# pylint: disable-next=too-many-arguments
def cli(
    model: Annotated[str, typer.Argument()],
    prompt_file: Annotated[
        Optional[Path], typer.Option(dir_okay=False, readable=True, resolve_path=True)
    ] = None,
    prompt: Annotated[
        Optional[str], typer.Option(help="a prompt to generate for")
    ] = None,
    output_folder: Annotated[
        Path, typer.Option(exists=False, resolve_path=True)
    ] = Path("."),
    iterations: Annotated[int, typer.Option()] = 30,
    img_per_prompt: Annotated[int, typer.Option()] = 3,
    loras: Annotated[
        Optional[list[str]], typer.Option("--lora", default_factory=list)
    ] = None,
) -> None:
    device = get_device()

    # pipe = StableDiffusionXLPipeline.from_pretrained(model, torch_dtype=torch.float16)
    pipe = DiffusionPipeline.from_pretrained(model, torch_dtype=torch.float16)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.safety_checker = safety_checker
    # pipe.enable_vae_tiling() #?
    pipe = pipe.to(device)

    if loras:
        for lora in loras:
            print(f"loading lora: {lora}")
            pipe.load_lora_weights(".", weight_name=lora)
            pipe.fuse_lora(lora_scale=0.7)

    os.makedirs(output_folder, exist_ok=True)
    print("generating...")

    generator = torch.Generator(device=device).manual_seed(0)

    if prompt_file:
        with open(prompt_file, encoding="utf8") as f:
            prompts = f.readlines()
    elif prompt:
        prompts = [prompt]

    for prompt_ in prompts:
        prompt_ = prompt_.strip()
        if prompt_.startswith("#") or prompt_ == "":
            continue
        print(f"generating for prompt: {prompt_}")
        filename = get_prompt_id(prompt_)
        print(filename)
        images = pipe(
            prompt_,
            num_inference_steps=iterations,
            # negative_prompt=neg_prompt,
            num_images_per_prompt=img_per_prompt,
            output_type="pil",
            generator=generator,
        ).images

        print(f"saving to {output_folder}")
        for i, image in enumerate(images):
            image.save(output_folder / f"{filename}_{i}.png")


# pylint: disable-next=unused-argument
def safety_checker(images):
    return images, [False] * len(images)


def get_prompt_id(prompt: str) -> str:
    hash_ = b64encode(bytes(str(hash(prompt)), "ascii")).decode("utf-8")
    first = prompt[0:8]
    first = re.sub(r"\W+", "_", first)
    # initials = "".join([word[0] for word in prompt.split()])
    # return f"{first}_{initials}_{hash_}"
    return f"{first}_{hash_}"


def cli2() -> None:
    typer.run(cli)


if __name__ == "__main__":
    cli2()
