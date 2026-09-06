import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

import typer
from PIL import Image
from tqdm import tqdm

from vlm_tools.bbox import parse_bboxes

if TYPE_CHECKING:
    from vlm_tools.logo import LogoRemover

app = typer.Typer()


class RemovalMethod(str, Enum):
    """Logo removal methods."""

    GRAY = "gray"
    OPENCV = "opencv"


def process_single_image(
    image_path: Path, bboxes: list, output_path: Path, method: RemovalMethod
) -> tuple[bool, str, Path | str]:
    """Process a single image (used for parallel processing)."""
    # deferred so `vlm --help` works without the logo extra installed
    from vlm_tools.logo import GrayLogoRemover, OpenCVLogoRemover

    try:
        remover: "LogoRemover"
        if method == RemovalMethod.OPENCV:
            remover = OpenCVLogoRemover()
        else:
            remover = GrayLogoRemover()

        image = Image.open(image_path).convert("RGB")
        result = remover.remove(image, bboxes)
        result.save(output_path)

        return (True, image_path.name, output_path)
    except Exception as e:
        return (False, image_path.name, str(e))


@app.command()
def main(
    bboxes: Path = typer.Argument(
        ..., help="JSON file with bounding boxes (output from vlm-process)"
    ),
    output_dir: Path = typer.Argument(None, help="Output directory for cleaned images"),
    method: RemovalMethod = typer.Option(
        RemovalMethod.OPENCV, help="Inpainting method"
    ),
) -> None:
    """Remove logos from images using inpainting.

    Expects a JSON file where each entry has an 'output' field containing
    bounding boxes in the format:
    [{"bbox_2d": [x1, y1, x2, y2], "label": "logo"}]

    Methods:
    - gray: Replace with gray rectangles (instant, simple)
    - opencv: Fast, lightweight inpainting using OpenCV (recommended for simple cases)
    """

    if not bboxes.exists():
        print(f"Error: {bboxes} does not exist", flush=True)
        raise typer.Exit(1)

    # default output directory to same location as input JSON
    if output_dir is None:
        output_dir = bboxes.parent / f"{bboxes.stem}_cleaned"

    output_dir.mkdir(exist_ok=True, parents=True)

    # load detections from JSON lines file
    detections = []
    json_dir = bboxes.parent
    with open(bboxes) as f:
        for line in f:
            if line.strip():
                detections.append(json.loads(line))

    print(f"Loaded {len(detections)} detections", flush=True)

    # parse bboxes and filter out images without any
    images_to_clean = []
    for detection in detections:
        bboxes_list = parse_bboxes(detection["output"])
        if bboxes_list:
            detection["bboxes"] = bboxes_list
            # resolve relative path
            rel_path = Path(detection["file_name"])
            detection["abs_path"] = (json_dir / rel_path).resolve()
            images_to_clean.append(detection)

    print(f"Found {len(images_to_clean)} images with logos to remove", flush=True)

    # prepare tasks for parallel processing
    tasks = []
    for detection in images_to_clean:
        image_path = detection["abs_path"]
        bbox_list = detection["bboxes"]
        output_path = output_dir / f"{image_path.name}"
        tasks.append((image_path, bbox_list, output_path, method))

    # process images in parallel
    num_workers = os.cpu_count()
    print(
        f"Processing {len(images_to_clean)} images with {num_workers} workers...",
        flush=True,
    )

    results = []
    errors = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(
                process_single_image, img_path, bbox_list, out_path, method
            ): (
                img_path,
                len(bbox_list),
            )
            for img_path, bbox_list, out_path, method in tasks
        }

        # process results as they complete with progress bar
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Processing"
        ):
            img_path, num_bboxes = futures[future]
            success, name, result_or_error = future.result()

            if success:
                results.append(result_or_error)
            else:
                errors.append((name, result_or_error))

    # report any errors
    if errors:
        print(f"\nErrors encountered:", flush=True)
        for name, error in errors:
            print(f"  ✗ {name}: {error}", flush=True)

    print(f"Processed {len(results)}/{len(images_to_clean)} images", flush=True)
    print(f"Cleaned images saved to: {output_dir}", flush=True)


if __name__ == "__main__":
    app()
