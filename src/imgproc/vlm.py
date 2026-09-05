from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from PIL import Image
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BatchFeature,
    PreTrainedModel,
    ProcessorMixin,
)

from imgproc.img_utils import resize_image_if_needed


def _load_image(path: Path, max_dim: int | None) -> Image.Image | None:
    """Load and optionally resize a single image, returning None on failure."""
    try:
        image: Image.Image = Image.open(path)
        image.load()
        if max_dim is not None:
            image = resize_image_if_needed(image, max_size=max_dim)
        else:
            image = resize_image_if_needed(image)
        return image
    except OSError as e:
        print(f"Skipping corrupted image {path}: {e}")
        return None


def prepare_vlm_batch(
    processor: ProcessorMixin,
    image_paths: list[Path],
    prompt: str,
    device: str | None,
    max_dim: int | None = None,
) -> tuple[BatchFeature | None, list[Path]]:
    """Prepare a batch of images for VLM inference."""
    with ThreadPoolExecutor(max_workers=len(image_paths)) as pool:
        images = list(pool.map(lambda p: _load_image(p, max_dim), image_paths))

    all_messages = []
    valid_paths = []
    valid_images = []
    for path, image in zip(image_paths, images):
        if image is None:
            continue
        all_messages.append(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
        )
        valid_paths.append(path)
        valid_images.append(image)

    if not all_messages:
        return None, []

    texts = [
        # transformers types content as plain str, but multimodal messages nest a list
        processor.apply_chat_template(
            msgs,  # type: ignore[arg-type]
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for msgs in all_messages
    ]

    # one image per text; the processor resizes to its own patch multiple
    batched_images = [[img] for img in valid_images]
    inputs = processor(
        text=texts, images=batched_images, padding=True, return_tensors="pt"
    )

    if device:
        inputs = inputs.to(device)

    return inputs, valid_paths


def load_model(model_name: str, device: str) -> tuple[PreTrainedModel, ProcessorMixin]:
    """Load VLM and processor."""
    model = AutoModelForImageTextToText.from_pretrained(
        model_name, device_map=device, dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    processor = AutoProcessor.from_pretrained(model_name)
    # set left padding for decoder-only models
    processor.tokenizer.padding_side = "left"
    return model, processor
