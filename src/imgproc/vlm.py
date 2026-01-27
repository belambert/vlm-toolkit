from pathlib import Path

import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForImageTextToText, AutoProcessor

from imgproc.img_utils import resize_image_if_needed


def prepare_vlm_batch(
    processor,
    image_paths: list[Path],
    prompt: str,
    device: str,
    max_dim: int | None = None,
):
    """Prepare a batch of images for VLM inference.

    Args:
        processor: The VLM processor
        image_paths: List of paths to images
        prompt: The text prompt to use for each image
        device: Device to move tensors to
        max_dim: Maximum dimension for image resizing (default: 1024)

    Returns:
        Processed inputs ready for model.generate()
    """
    # Load and prepare all images
    all_messages = []
    for image_path in image_paths:
        image = Image.open(image_path)
        if max_dim is not None:
            image = resize_image_if_needed(image, max_size=max_dim)
        else:
            image = resize_image_if_needed(image)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        all_messages.append(messages)

    # Prepare batch inputs
    texts = []
    all_image_inputs = []
    all_video_inputs = []

    for messages in all_messages:
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        texts.append(text)

        image_inputs, video_inputs = process_vision_info(messages)
        if image_inputs is not None:
            all_image_inputs.extend(image_inputs)
        if video_inputs is not None:
            all_video_inputs.extend(video_inputs)

    # Process batch
    inputs = processor(
        text=texts,
        images=all_image_inputs if all_image_inputs else None,
        videos=all_video_inputs if all_video_inputs else None,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(device)

    return inputs


def load_model(model_name: str, device: str):
    """Load VLM and processor."""
    model = AutoModelForImageTextToText.from_pretrained(
        model_name, device_map=device, dtype=torch.bfloat16
    )
    processor = AutoProcessor.from_pretrained(model_name)
    # Set left padding for decoder-only models
    processor.tokenizer.padding_side = "left"
    return model, processor
