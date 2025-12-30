import json
from pathlib import Path

import torch
import typer
from diffusers import AutoPipelineForInpainting
from PIL import Image, ImageDraw

from imggen.util import get_device

app = typer.Typer()


def create_mask_for_location(width: int, height: int, location: str) -> Image.Image:
    """Create a mask for the watermark location."""
    mask = Image.new("RGB", (width, height), color="black")
    draw = ImageDraw.Draw(mask)

    # Define mask size as percentage of image dimensions
    mask_width = int(width * 0.25)  # 25% of image width
    mask_height = int(height * 0.15)  # 15% of image height

    # Determine coordinates based on location
    if location == "top-left":
        x0, y0 = 0, 0
        x1, y1 = mask_width, mask_height
    elif location == "top-right":
        x0, y0 = width - mask_width, 0
        x1, y1 = width, mask_height
    elif location == "bottom-left":
        x0, y0 = 0, height - mask_height
        x1, y1 = mask_width, height
    elif location == "bottom-right":
        x0, y0 = width - mask_width, height - mask_height
        x1, y1 = width, height
    else:
        # Unknown location, return empty mask
        return mask

    # Draw white rectangle where watermark is
    draw.rectangle([x0, y0, x1, y1], fill="white")

    return mask


def remove_watermark(
    pipe, image_path: Path, location: str, output_dir: Path, device: str
) -> Path:
    """Remove watermark from an image using inpainting."""
    # Load image
    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    # Create mask
    mask = create_mask_for_location(width, height, location)

    # Run inpainting
    result = pipe(
        prompt="clean background, no watermark, no text",
        negative_prompt="watermark, logo, text, signature",
        image=image,
        mask_image=mask,
        num_inference_steps=20,
        guidance_scale=7.5,
        width=width,
        height=height,
    ).images[0]

    # Save result
    output_path = output_dir / f"cleaned_{image_path.name}"
    result.save(output_path)

    return output_path


@app.command()
def main(
    detections_json: Path = typer.Argument(
        ..., help="JSON file from detect_logo.py script"
    ),
    output_dir: Path = typer.Option(
        None, help="Output directory for cleaned images"
    ),
    model_name: str = typer.Option(
        "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        help="Inpainting model to use",
    ),
):
    """Remove watermarks from images using inpainting."""

    if not detections_json.exists():
        typer.echo(f"Error: {detections_json} does not exist")
        raise typer.Exit(1)

    # Set output directory
    if output_dir is None:
        output_dir = detections_json.parent / "cleaned"
    output_dir.mkdir(exist_ok=True, parents=True)

    # Load detections
    with open(detections_json) as f:
        detections = json.load(f)

    typer.echo(f"Loaded {len(detections)} detections")

    # Filter out images without watermarks
    images_to_clean = [
        d for d in detections if "none" not in d["detection"].lower()
    ]

    if not images_to_clean:
        typer.echo("No watermarks to remove!")
        raise typer.Exit(0)

    typer.echo(f"Found {len(images_to_clean)} images with watermarks to remove")

    # Load model
    device = get_device()
    typer.echo(f"Using device: {device}")
    typer.echo(f"Loading model: {model_name}...")

    pipe = AutoPipelineForInpainting.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
    )
    pipe.to(device)

    typer.echo("Model loaded successfully\n")

    # Process each image
    results = []
    for detection in images_to_clean:
        image_path = Path(detection["image_path"])
        location = detection["detection"].strip()

        typer.echo(f"Processing: {image_path.name} ({location})...", nl=False)

        try:
            output_path = remove_watermark(
                pipe, image_path, location, output_dir, device
            )
            results.append(output_path)
            typer.echo(f" ✓ Saved to {output_path.name}")
        except Exception as e:
            typer.echo(f" ✗ Error: {e}")
            continue

    typer.echo(f"\nProcessed {len(results)}/{len(images_to_clean)} images")
    typer.echo(f"Cleaned images saved to: {output_dir}")


if __name__ == "__main__":
    app()
