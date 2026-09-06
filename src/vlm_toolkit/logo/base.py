"""Base class for logo removal algorithms."""

from abc import ABC, abstractmethod

from PIL import Image


class LogoRemover(ABC):
    """Base class for logo removal algorithms."""

    @abstractmethod
    def remove(self, image: Image.Image, bboxes: list[dict]) -> Image.Image:
        """Remove the logos at the given bbox_2d regions from an image."""
        pass
