from pathlib import Path

from PIL import Image

IMG_EXTENSIONS = ["jpg", "jpeg", "png", "webp"]


def resize_image_if_needed(image: Image.Image, max_size: int = 1024) -> Image.Image:
    """Resize image if larger than max_size, maintaining aspect ratio."""
    width, height = image.size

    if width > max_size or height > max_size:
        # calculate new size maintaining aspect ratio
        if width > height:
            new_width = max_size
            new_height = int(height * (max_size / width))
        else:
            new_height = max_size
            new_width = int(width * (max_size / height))

        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    return image


def find_images(folder: Path) -> list[Path]:
    """Find all images in a folder based on IMG_EXTENSIONS."""
    image_files: list[Path] = []
    for ext in IMG_EXTENSIONS:
        image_files.extend(folder.glob(f"*.{ext}"))
    print(f"Found {len(image_files):,} images to check", flush=True)
    return image_files


def ensure_dimensions_divisible_by_8(image: Image.Image) -> Image.Image:
    """Crop pixels from the edges so both dimensions are divisible by 8."""
    orig_width, orig_height = image.size
    width = (orig_width // 8) * 8
    height = (orig_height // 8) * 8

    if width != orig_width or height != orig_height:
        # calculate pixels to remove from each edge
        width_diff = orig_width - width
        height_diff = orig_height - height

        # remove evenly from both sides (if odd, remove extra from right/bottom)
        left = width_diff // 2
        top = height_diff // 2
        right = orig_width - (width_diff - left)
        bottom = orig_height - (height_diff - top)

        image = image.crop((left, top, right, bottom))

    return image
