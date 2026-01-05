import json
import re
from pathlib import Path

import cv2
import numpy as np
import torch
import typer
from diffusers import AutoPipelineForInpainting
from PIL import Image, ImageDraw

from imggen.util import get_device

app = typer.Typer()


def parse_bboxes(output: str) -> list[dict]:
    """Parse bounding boxes from VLM output.

    Expects output to be JSON (possibly wrapped in markdown code blocks)
    containing an array of objects with bbox_2d and label fields.
    """
    # Strip markdown code blocks if present
    output = output.strip()
    if output.startswith("```"):
        # Extract content between code blocks
        match = re.search(r"```(?:json)?\s*\n(.*?)\n```", output, re.DOTALL)
        if match:
            output = match.group(1)

    try:
        bboxes = json.loads(output)
        return bboxes if isinstance(bboxes, list) else []
    except json.JSONDecodeError:
        return []


def scale_bbox(
    bbox_coords: list[int], width: int, height: int
) -> tuple[int, int, int, int]:
    """Scale bounding box from normalized 1000x1000 coordinates to actual image size.

    Args:
        bbox_coords: [x1, y1, x2, y2] in range [0, 1000]
        width: Actual image width in pixels
        height: Actual image height in pixels

    Returns:
        Scaled bbox coordinates (x1, y1, x2, y2) in pixels, with 15% buffer added
    """
    x1, y1, x2, y2 = bbox_coords
    # Convert from 1000-based coordinates to actual pixel coordinates
    scaled_x1 = x1 * width / 1000
    scaled_y1 = y1 * height / 1000
    scaled_x2 = x2 * width / 1000
    scaled_y2 = y2 * height / 1000

    # Add 15% buffer in all directions
    bbox_width = scaled_x2 - scaled_x1
    bbox_height = scaled_y2 - scaled_y1

    buffer_x = bbox_width * 0.15
    buffer_y = bbox_height * 0.15

    # Expand bbox and clamp to image boundaries
    scaled_x1 = max(0, int(scaled_x1 - buffer_x))
    scaled_y1 = max(0, int(scaled_y1 - buffer_y))
    scaled_x2 = min(width, int(scaled_x2 + buffer_x))
    scaled_y2 = min(height, int(scaled_y2 + buffer_y))

    return scaled_x1, scaled_y1, scaled_x2, scaled_y2


def create_mask_from_bboxes(width: int, height: int, bboxes: list[dict]) -> Image.Image:
    """Create a mask from bounding boxes."""
    mask = Image.new("RGB", (width, height), color="black")
    draw = ImageDraw.Draw(mask)

    for bbox in bboxes:
        if "bbox_2d" in bbox:
            # Scale bbox from 1000x1000 to actual image dimensions
            x1, y1, x2, y2 = scale_bbox(bbox["bbox_2d"], width, height)
            # Draw white rectangle where logo is
            draw.rectangle([x1, y1, x2, y2], fill="white")

    return mask


def ensure_dimensions_divisible_by_8(image: Image.Image) -> Image.Image:
    """Ensure image dimensions are divisible by 8 by cropping pixels from edges.

    Args:
        image: PIL Image to check/crop

    Returns:
        Cropped image with dimensions divisible by 8
    """
    orig_width, orig_height = image.size
    width = (orig_width // 8) * 8
    height = (orig_height // 8) * 8

    if width != orig_width or height != orig_height:
        # Calculate pixels to remove from each edge
        width_diff = orig_width - width
        height_diff = orig_height - height

        # Remove evenly from both sides (if odd, remove extra from right/bottom)
        left = width_diff // 2
        top = height_diff // 2
        right = orig_width - (width_diff - left)
        bottom = orig_height - (height_diff - top)

        image = image.crop((left, top, right, bottom))

    return image


def remove_logo_gray(
    image_path: Path, bboxes: list[dict], output_dir: Path
) -> Path:
    """Remove logo from an image by replacing with gray rectangles."""
    # Load image with OpenCV
    image = cv2.imread(str(image_path))
    height, width = image.shape[:2]

    # Draw gray rectangles over bboxes
    for bbox in bboxes:
        if "bbox_2d" in bbox:
            # Scale bbox from 1000x1000 to actual image dimensions
            x1, y1, x2, y2 = scale_bbox(bbox["bbox_2d"], width, height)
            # Fill with medium gray (128, 128, 128)
            cv2.rectangle(image, (x1, y1), (x2, y2), (128, 128, 128), -1)

    # Save result
    output_path = output_dir / f"{image_path.name}"
    cv2.imwrite(str(output_path), image)

    return output_path


def remove_logo_opencv(
    image_path: Path, bboxes: list[dict], output_dir: Path
) -> Path:
    """Remove logo from an image using OpenCV inpainting."""
    # Load image with OpenCV
    image = cv2.imread(str(image_path))
    height, width = image.shape[:2]

    # Create binary mask
    mask = np.zeros((height, width), dtype=np.uint8)
    for bbox in bboxes:
        if "bbox_2d" in bbox:
            # Scale bbox from 1000x1000 to actual image dimensions
            x1, y1, x2, y2 = scale_bbox(bbox["bbox_2d"], width, height)
            # Draw white rectangle where logo is
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

    # Run OpenCV inpainting (Telea method)
    result = cv2.inpaint(image, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

    # Save result
    output_path = output_dir / f"{image_path.name}"
    cv2.imwrite(str(output_path), result)

    return output_path


def remove_logo_sdxl(
    pipe, image_path: Path, bboxes: list[dict], output_dir: Path, device: str
) -> Path:
    """Remove logo from an image using SDXL inpainting."""
    # Load image
    image = Image.open(image_path).convert("RGB")

    # Ensure dimensions are divisible by 8
    image = ensure_dimensions_divisible_by_8(image)
    width, height = image.size

    # Create mask from bounding boxes
    mask = create_mask_from_bboxes(width, height, bboxes)

    # Run inpainting
    result = pipe(
        prompt="clean background, no logo, no text",
        negative_prompt="logo, text, signature",
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
        ..., help="JSON file with bounding boxes (output from vlm-process)"
    ),
    output_dir: Path = typer.Argument(None, help="Output directory for cleaned images"),
    method: str = typer.Option(
        "opencv", help="Inpainting method: 'gray', 'opencv', or 'sdxl'"
    ),
    model_name: str = typer.Option(
        "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        help="Inpainting model to use (only for sdxl method)",
    ),
):
    """Remove logos from images using inpainting.

    Expects a JSON file where each entry has an 'output' field containing
    bounding boxes in the format:
    [{"bbox_2d": [x1, y1, x2, y2], "label": "logo"}]

    Methods:
    - gray: Replace with gray rectangles (instant, simple)
    - opencv: Fast, lightweight inpainting using OpenCV (recommended for simple cases)
    - sdxl: High-quality inpainting using Stable Diffusion XL (slower, better quality)
    """

    if method not in ["gray", "opencv", "sdxl"]:
        print(
            f"Error: Invalid method '{method}'. Choose 'gray', 'opencv', or 'sdxl'",
            flush=True,
        )
        raise typer.Exit(1)

    if not detections_json.exists():
        print(f"Error: {detections_json} does not exist", flush=True)
        raise typer.Exit(1)

    output_dir.mkdir(exist_ok=True, parents=True)

    # Load detections
    with open(detections_json) as f:
        detections = json.load(f)

    print(f"Loaded {len(detections)} detections", flush=True)

    # Parse bboxes and filter out images without any
    images_to_clean = []
    for detection in detections:
        bboxes = parse_bboxes(detection["output"])
        if bboxes:
            detection["bboxes"] = bboxes
            images_to_clean.append(detection)

    print(f"Found {len(images_to_clean)} images with logos to remove", flush=True)

    # Load model if using SDXL
    pipe = None
    device = None
    if method == "sdxl":
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
    elif method == "opencv":
        print(f"Using OpenCV inpainting method\n", flush=True)
    else:
        print(f"Using gray rectangle replacement method\n", flush=True)

    # Process each image
    results = []
    for detection in images_to_clean:
        image_path = Path(detection["image_path"])
        bboxes = detection["bboxes"]
        num_bboxes = len(bboxes)

        print(
            f"Processing: {image_path.name} ({num_bboxes} bbox{'es' if num_bboxes > 1 else ''})...",
            end="",
            flush=True,
        )

        try:
            if method == "gray":
                output_path = remove_logo_gray(image_path, bboxes, output_dir)
            elif method == "opencv":
                output_path = remove_logo_opencv(image_path, bboxes, output_dir)
            else:
                output_path = remove_logo_sdxl(
                    pipe, image_path, bboxes, output_dir, device
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
