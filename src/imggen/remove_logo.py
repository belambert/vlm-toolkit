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
    orig_width, orig_height = image.size

    # Ensure dimensions are divisible by 8
    width = (orig_width // 8) * 8
    height = (orig_height // 8) * 8

    # Resize if needed
    if width != orig_width or height != orig_height:
        image = image.resize((width, height), Image.Resampling.LANCZOS)

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
        # strength=1.0,
        width=width,
        height=height,
    ).images[0]

    # Save result
    output_path = output_dir / f"{image_path.name}"
    result.save(output_path)

    return output_path


@app.command()
def main(
    detections_json: Path = typer.Argument(
        ..., help="JSON file from vlm-process or detect-logo"
    ),
    output_dir: Path = typer.Argument(None, help="Output directory for cleaned images"),
    model_name: str = typer.Option(
        "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        help="Inpainting model to use",
    ),
):
    """Remove watermarks from images using inpainting."""

    if not detections_json.exists():
        print(f"Error: {detections_json} does not exist", flush=True)
        raise typer.Exit(1)

    output_dir.mkdir(exist_ok=True, parents=True)

    # Load detections
    with open(detections_json) as f:
        detections = json.load(f)

    print(f"Loaded {len(detections)} detections", flush=True)

    # Filter out images without watermarks
    images_to_clean = [d for d in detections if "none" not in d["output"].lower()]

    print(f"Found {len(images_to_clean)} images with watermarks to remove", flush=True)

    # Load model
    device = get_device()
    print(f"Using device: {device}", flush=True)
    print(f"Loading model: {model_name}...", flush=True)

    pipe = AutoPipelineForInpainting.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe.set_progress_bar_config(disable=True)
    pipe.to(device)

    # Enable memory optimizations
    pipe.enable_attention_slicing()
    pipe.vae.enable_slicing()

    print("Model loaded successfully\n", flush=True)

    # Process each image
    results = []
    for detection in images_to_clean:
        image_path = Path(detection["image_path"])
        location = detection["output"].strip()

        print(f"Processing: {image_path.name} ({location})...", end="", flush=True)

        try:
            output_path = remove_watermark(
                pipe, image_path, location, output_dir, device
            )
            results.append(output_path)
            print(f" ✓ Saved to {output_path.name}", flush=True)
        except Exception as e:
            print(f" ✗ Error: {e}", flush=True)
            continue

    print(f"\nProcessed {len(results)}/{len(images_to_clean)} images", flush=True)
    print(f"Cleaned images saved to: {output_dir}", flush=True)


if __name__ == "__main__":
    app()
