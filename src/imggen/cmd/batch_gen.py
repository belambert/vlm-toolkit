import json
from pathlib import Path

import torch
import typer
import yaml
from diffusers import DiffusionPipeline

from imggen.util import get_device

app = typer.Typer()


def load_prompts(prompts_file: Path) -> list[str]:
    """Load prompts from JSON lines or YAML file."""
    if prompts_file.suffix in [".yaml", ".yml"]:
        with open(prompts_file) as f:
            return yaml.safe_load(f)
    else:
        # JSON lines format
        with open(prompts_file) as f:
            lines = f.readlines()
        return [json.loads(line)["prompt"] for line in lines]


@app.command()
def main(
    prompts_file: Path = typer.Argument(
        ..., help="JSON lines or YAML file with prompts"
    ),
    output_dir: Path = typer.Argument(..., help="Output directory"),
    model: str = typer.Option(
        "stabilityai/stable-diffusion-xl-base-1.0",
        help="Model to use for image generation",
    ),
    num_inference_steps: int = typer.Option(25, help="Number of denoising steps"),
    images_per_prompt: int = typer.Option(
        4, help="Number of images to generate per prompt"
    ),
    size: int = typer.Option(1024, help="Image size (width and height)"),
):
    """
    Suggested models:
    - stabilityai/stable-diffusion-xl-base-1.0
    - black-forest-labs/FLUX.1-dev
    - black-forest-labs/FLUX.2-dev

    """
    output_dir.mkdir(exist_ok=True, parents=True)

    # Load prompts
    prompts = load_prompts(prompts_file)
    print(f"Loaded {len(prompts)} prompts", flush=True)

    # Load model
    pipe = DiffusionPipeline.from_pretrained(model, dtype=torch.bfloat16)
    pipe.to(get_device())

    # Generate images
    total_images = 0
    for i, prompt in enumerate(prompts):
        print(f"Generating prompt {i+1}/{len(prompts)}: {prompt}", flush=True)
        images = pipe(
            prompt,
            num_inference_steps=num_inference_steps,
            num_images_per_prompt=images_per_prompt,
            height=size,
            width=size,
        ).images
        for j, image in enumerate(images):
            image.save(output_dir / f"image_{i:04d}_{j:04d}.png")
            total_images += 1

    print(f"\nGenerated {total_images} images in {output_dir}", flush=True)


if __name__ == "__main__":
    app()
