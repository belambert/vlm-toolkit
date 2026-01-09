"""SDXL inpainting logo removal implementation."""

import torch
from diffusers import AutoPipelineForInpainting
from PIL import Image

from imggen.bbox import create_mask_from_bboxes
from imggen.img_utils import ensure_dimensions_divisible_by_8
from imggen.logo.base import LogoRemover
from imggen.util import get_device


class SDXLLogoRemover(LogoRemover):
    """Remove logos using SDXL inpainting."""

    def __init__(self, model_name: str):
        """Initialize with SDXL inpainting model.

        Args:
            model_name: Name of the inpainting model to load
        """
        device = get_device()
        print(f"Using device: {device}", flush=True)
        print(f"Loading model: {model_name}...", flush=True)

        self.pipe = AutoPipelineForInpainting.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            safety_checker=None,
            requires_safety_checker=False,
        )
        self.pipe.set_progress_bar_config(disable=True)
        self.pipe.to(device)

        # Enable memory optimizations
        self.pipe.enable_attention_slicing()
        self.pipe.vae.enable_slicing()

        print("Model loaded successfully\n", flush=True)

    def remove(self, image: Image.Image, bboxes: list[dict]) -> Image.Image:
        """Remove logo from an image using SDXL inpainting."""
        # Ensure dimensions are divisible by 8
        image = ensure_dimensions_divisible_by_8(image)
        width, height = image.size

        # Create mask from bounding boxes
        mask = create_mask_from_bboxes(width, height, bboxes)

        # Run inpainting
        result = self.pipe(
            prompt="clean background, no logo, no text",
            negative_prompt="logo, text, signature",
            image=image,
            mask_image=mask,
            num_inference_steps=20,
            guidance_scale=7.5,
            # strength=1.0,
            width=width,
            height=height,
        ).images[0]

        return result
