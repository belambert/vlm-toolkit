import json
from pathlib import Path

import typer
from PIL import Image

from imggen.bbox import parse_bboxes
from imggen.logo import GrayLogoRemover, LogoRemover, OpenCVLogoRemover, SDXLLogoRemover

app = typer.Typer()


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
        remover = SDXLLogoRemover(model_name)
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
