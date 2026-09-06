import json
import re

from PIL import Image, ImageDraw


def parse_bboxes(output: str) -> list[dict]:
    """Parse bbox_2d/label objects from VLM output, unwrapping markdown code blocks."""
    # strip markdown code blocks if present
    output = output.strip()
    if output.startswith("```"):
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
    """Scale a bbox from 1000x1000 coords to pixels, adding a 15% buffer."""
    x1, y1, x2, y2 = bbox_coords
    # convert from 1000-based coordinates to actual pixel coordinates
    scaled_x1 = x1 * width / 1000
    scaled_y1 = y1 * height / 1000
    scaled_x2 = x2 * width / 1000
    scaled_y2 = y2 * height / 1000

    # add 15% buffer in all directions
    bbox_width = scaled_x2 - scaled_x1
    bbox_height = scaled_y2 - scaled_y1

    buffer_x = bbox_width * 0.15
    buffer_y = bbox_height * 0.15

    # expand bbox and clamp to image boundaries
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
            # scale bbox from 1000x1000 to actual image dimensions
            x1, y1, x2, y2 = scale_bbox(bbox["bbox_2d"], width, height)
            # draw white rectangle where logo is
            draw.rectangle([x1, y1, x2, y2], fill="white")

    return mask
