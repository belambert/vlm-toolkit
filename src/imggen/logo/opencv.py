"""OpenCV inpainting logo removal implementation."""

import cv2
import numpy as np
from PIL import Image

from imggen.bbox import scale_bbox
from imggen.logo.base import LogoRemover


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
