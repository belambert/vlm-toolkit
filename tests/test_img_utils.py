from PIL import Image

from imggen.img_utils import resize_image_if_needed


def test_resize_image_if_needed_no_resize():
    """Test that small images are not resized."""
    # Create a small image
    img = Image.new("RGB", (500, 500))
    resized = resize_image_if_needed(img, max_size=1024)

    assert resized.size == (500, 500)


def test_resize_image_if_needed_width_larger():
    """Test resizing when width is larger than max_size."""
    # Create a wide image
    img = Image.new("RGB", (2000, 1000))
    resized = resize_image_if_needed(img, max_size=1024)

    assert resized.size[0] == 1024
    assert resized.size[1] == 512  # Maintains aspect ratio


def test_resize_image_if_needed_height_larger():
    """Test resizing when height is larger than max_size."""
    # Create a tall image
    img = Image.new("RGB", (1000, 2000))
    resized = resize_image_if_needed(img, max_size=1024)

    assert resized.size[0] == 512  # Maintains aspect ratio
    assert resized.size[1] == 1024
