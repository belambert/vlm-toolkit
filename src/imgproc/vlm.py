from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from imgproc.img_utils import resize_image_if_needed


def _load_image(path: Path, max_dim: int | None) -> Image.Image | None:
    """Load and optionally resize a single image, returning None on failure."""
    try:
        image = Image.open(path)
        image.load()
        if max_dim is not None:
            image = resize_image_if_needed(image, max_size=max_dim)
        else:
            image = resize_image_if_needed(image)
        return image
    except OSError as e:
        print(f"Skipping corrupted image {path}: {e}")
        return None


def _is_qwen_processor(processor) -> bool:
    return "qwen" in type(processor).__name__.lower()


def _build_image_inputs(processor, all_messages, valid_images):
    """Build processor kwargs for images based on model type."""
    if _is_qwen_processor(processor):
        # qwen uses process_vision_info for flat image/video lists
        from qwen_vl_utils import process_vision_info

        all_image_inputs, all_video_inputs = [], []
        for msgs in all_messages:
            img_in, vid_in = process_vision_info(msgs)
            if img_in:
                all_image_inputs.extend(img_in)
            if vid_in:
                all_video_inputs.extend(vid_in)
        return {
            "images": all_image_inputs or None,
            "videos": all_video_inputs or None,
        }

    # other models (gemma 4, etc.) expect per-text batched images
    return {"images": [[img] for img in valid_images]}


def prepare_vlm_batch(
    processor,
    image_paths: list[Path],
    prompt: str,
    device: str,
    max_dim: int | None = None,
):
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
        processor.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        for msgs in all_messages
    ]

    image_kwargs = _build_image_inputs(processor, all_messages, valid_images)
    inputs = processor(text=texts, **image_kwargs, padding=True, return_tensors="pt")

    if device:
        inputs = inputs.to(device)

    return inputs, valid_paths


def load_model(model_name: str, device: str):
    """Load VLM and processor."""
    model = AutoModelForImageTextToText.from_pretrained(
        model_name, device_map=device, dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    processor = AutoProcessor.from_pretrained(model_name)
    # set left padding for decoder-only models
    processor.tokenizer.padding_side = "left"
    return model, processor
