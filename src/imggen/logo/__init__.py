"""Logo removal classes."""

from imggen.logo.base import LogoRemover
from imggen.logo.gray import GrayLogoRemover
from imggen.logo.opencv import OpenCVLogoRemover
from imggen.logo.sdxl import SDXLLogoRemover

__all__ = ["LogoRemover", "GrayLogoRemover", "OpenCVLogoRemover", "SDXLLogoRemover"]
