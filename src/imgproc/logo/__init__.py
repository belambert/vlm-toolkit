"""Logo removal classes."""

from imgproc.logo.base import LogoRemover
from imgproc.logo.gray import GrayLogoRemover
from imgproc.logo.opencv import OpenCVLogoRemover

__all__ = ["LogoRemover", "GrayLogoRemover", "OpenCVLogoRemover"]
