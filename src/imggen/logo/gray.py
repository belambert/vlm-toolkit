"""Gray rectangle logo removal implementation."""

import cv2
import numpy as np
from PIL import Image

from imggen.bbox import scale_bbox
from imggen.logo.base import LogoRemover


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
