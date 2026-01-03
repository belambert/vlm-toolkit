from pathlib import Path

from PIL import Image

IMG_EXTENSIONS = ["jpg", "jpeg", "png", "webp"]


def resize_image_if_needed(image: Image.Image, max_size: int = 1024) -> Image.Image:
    """Resize image if larger than max_size, maintaining aspect ratio."""
    width, height = image.size

    if width > max_size or height > max_size:
        # Calculate new size maintaining aspect ratio
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
    image_files = []
    for ext in IMG_EXTENSIONS:
        image_files.extend(folder.glob(f"*.{ext}"))
    print(f"Found {len(image_files)} images to check", flush=True)
    return image_files
