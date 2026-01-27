"""Base class for logo removal algorithms."""

from abc import ABC, abstractmethod

from PIL import Image


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
