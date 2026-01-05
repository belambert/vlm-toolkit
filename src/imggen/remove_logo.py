import json
import re
from abc import ABC, abstractmethod
from pathlib import Path

import cv2
import numpy as np
import torch
import typer
from diffusers import AutoPipelineForInpainting
from PIL import Image, ImageDraw

from imggen.img_utils import ensure_dimensions_divisible_by_8
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


class LogoRemover(ABC):
    """Base class for logo removal algorithms."""

    @abstractmethod
    def remove(self, image: Image.Image, bboxes: list[dict]) -> Image.Image:
        """Remove logos from an image.

        Args:
            image: PIL Image to process
            bboxes: List of bounding box dictionaries with bbox_2d field

        Returns:
            Processed PIL Image with logos removed
        """
        pass


class GrayLogoRemover(LogoRemover):
    """Remove logos by replacing with gray rectangles."""

    def remove(self, image: Image.Image, bboxes: list[dict]) -> Image.Image:
        """Remove logo from an image by replacing with gray rectangles."""
        # Convert PIL Image to OpenCV format (BGR)
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        height, width = image_cv.shape[:2]

        # Draw gray rectangles over bboxes
        for bbox in bboxes:
            if "bbox_2d" in bbox:
                # Scale bbox from 1000x1000 to actual image dimensions
                x1, y1, x2, y2 = scale_bbox(bbox["bbox_2d"], width, height)
                # Fill with medium gray (128, 128, 128)
                cv2.rectangle(image_cv, (x1, y1), (x2, y2), (128, 128, 128), -1)

        # Convert back to PIL Image
        result = Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))
        return result


class OpenCVLogoRemover(LogoRemover):
    """Remove logos using OpenCV inpainting."""

    def remove(self, image: Image.Image, bboxes: list[dict]) -> Image.Image:
        """Remove logo from an image using OpenCV inpainting."""
        # Convert PIL Image to OpenCV format (BGR)
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        height, width = image_cv.shape[:2]

        # Create binary mask
        mask = np.zeros((height, width), dtype=np.uint8)
        for bbox in bboxes:
            if "bbox_2d" in bbox:
                # Scale bbox from 1000x1000 to actual image dimensions
                x1, y1, x2, y2 = scale_bbox(bbox["bbox_2d"], width, height)
                # Draw white rectangle where logo is
                cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

        # Run OpenCV inpainting (Telea method)
        result_cv = cv2.inpaint(
            image_cv, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA
        )

        # Convert back to PIL Image
        result = Image.fromarray(cv2.cvtColor(result_cv, cv2.COLOR_BGR2RGB))
        return result


class SDXLLogoRemover(LogoRemover):
    """Remove logos using SDXL inpainting."""

    def __init__(self, pipe):
        """Initialize with SDXL inpainting pipeline.

        Args:
            pipe: AutoPipelineForInpainting instance
        """
        self.pipe = pipe

    def remove(self, image: Image.Image, bboxes: list[dict]) -> Image.Image:
        """Remove logo from an image using SDXL inpainting."""
        # Ensure dimensions are divisible by 8
        image = ensure_dimensions_divisible_by_8(image)
        width, height = image.size

        # Create mask from bounding boxes
        mask = create_mask_from_bboxes(width, height, bboxes)

        # Run inpainting
        result = self.pipe(
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

        return result


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

    # Create logo remover instance based on selected method
    remover: LogoRemover
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
        remover = SDXLLogoRemover(pipe)
    elif method == "opencv":
        print(f"Using OpenCV inpainting method\n", flush=True)
        remover = OpenCVLogoRemover()
    else:
        print(f"Using gray rectangle replacement method\n", flush=True)
        remover = GrayLogoRemover()

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
            # Load image
            image = Image.open(image_path).convert("RGB")

            # Process image with remover
            result = remover.remove(image, bboxes)

            # Save result
            output_path = output_dir / f"{image_path.name}"
            result.save(output_path)

            results.append(output_path)
            print(f" ✓ Saved to {output_path.name}", flush=True)
        except Exception as e:
            print(f" ✗ Error: {e}", flush=True)
            continue

    print(f"\nProcessed {len(results)}/{len(images_to_clean)} images", flush=True)
    print(f"Cleaned images saved to: {output_dir}", flush=True)


if __name__ == "__main__":
    app()
