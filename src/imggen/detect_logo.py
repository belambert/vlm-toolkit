import json
from pathlib import Path

import torch
import typer
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForVision2Seq, AutoProcessor

from imggen.util import get_device

app = typer.Typer()

EXTENSIONS = ["jpg", "jpeg", "png", "webp"]

PROMPT = """Look at this image carefully. Does it contain any watermarks, logos, text overlays, or copyright marks?

If yes, output only the location:
 - top-left
 - top-right
 - bottom-left
 - bottom-right

If no watermark is present, just respond with "none"
"""


def detect_logo(model, processor, image_path: Path, device: str) -> dict:
    """Detect watermark in an image using Qwen2-VL."""
    # Load image and get original size
    image = Image.open(image_path)
    width, height = image.size

    # Resize if needed
    image = resize_image_if_needed(image)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": PROMPT},
            ],
        }
    ]

    # LOOK HERE TO UNDERSTAND THE MODEL BETTER

    # Prepare inputs
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    # print(text)
    image_inputs, video_inputs = process_vision_info(messages)
    # print(image_inputs)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    # would potentially do batching and padding here... or before applying the
    # tokenizer/processor
    inputs = inputs.to(device)
    # ['input_ids', 'attention_mask', 'pixel_values', 'image_grid_thw']
    # print(list(inputs.keys()))

    # input_ids
    # torch.Size([1, 967])
    # attention_mask
    # torch.Size([1, 967])
    # pixel_values
    # torch.Size([3552, 1176])
    # image_grid_thw
    # torch.Size([1, 3])
    # tensor([[ 1, 48, 74]], device='mps:0')

    # Generate detection
    generated_ids = model.generate(**inputs, max_new_tokens=512)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    response = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    return {"image_path": str(image_path), "detection": response}


def detect_logo_batch(model, processor, image_paths: list[Path], device: str) -> list[dict]:
    """Detect watermarks in a batch of images."""
    # Load and prepare all images
    all_messages = []
    for image_path in image_paths:
        image = Image.open(image_path)
        image = resize_image_if_needed(image)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": PROMPT},
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

    # Generate detections
    generated_ids = model.generate(**inputs, max_new_tokens=512)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    responses = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    # Create results
    results = []
    for image_path, response in zip(image_paths, responses):
        results.append({"image_path": str(image_path), "detection": response})

    return results


@app.command()
def main(
    folder: Path = typer.Argument(..., help="Folder containing images to check"),
    output: Path = typer.Option(None, help="Output JSON file"),
    model_name: str = typer.Option("Qwen/Qwen3-VL-8B-Instruct", help="HF model"),
    batch_size: int = typer.Option(1, help="Number of images to process in parallel"),
):
    """Detect logos with a VLM."""
    image_files = find_images(folder)
    device = get_device()
    model, processor = load_model(model_name, device)
    results = []

    # Process images in batches
    total_batches = (len(image_files) + batch_size - 1) // batch_size
    for i in range(0, len(image_files), batch_size):
        batch = image_files[i:i + batch_size]
        batch_num = i // batch_size + 1
        typer.echo(f"Processing batch {batch_num}/{total_batches} ({len(batch)} images)...")

        try:
            batch_results = detect_logo_batch(model, processor, batch, device)
            results.extend(batch_results)
        except Exception as e:
            typer.echo(f"  Error processing batch: {e}, skipping...")

    if output is None:
        output = folder / "logo_detections.json"

    with open(output, "w") as f:
        json.dump(results, f, indent=2)

    typer.echo(f"\nResults saved to: {output}")
    typer.echo(f"Processed: {len(results)}/{len(image_files)} images")


def find_images(folder: Path) -> list[Path]:
    """Find all images in a folder based on EXTENSIONS."""
    image_files = []
    for ext in EXTENSIONS:
        image_files.extend(folder.glob(f"*.{ext}"))
    typer.echo(f"Found {len(image_files)} images to check")
    return image_files


def resize_image_if_needed(image: Image.Image, max_size: int = 1024) -> Image.Image:
    """Resize image if larger than max_size, maintaining aspect ratio."""
    width, height = image.size

    if width > max_size or height > max_size:
        # Calculate new size maintaining aspect ratio
        if width > height:
            new_width = max_size
            new_height = int(height * (max_size / width))
        else:
            new_height = max_size
            new_width = int(width * (max_size / height))

        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    return image


def load_model(model_name: str, device: str):
    """Load VLM and processor."""
    model = AutoModelForVision2Seq.from_pretrained(
        model_name,
        dtype=torch.bfloat16 if device != "cpu" else torch.float32,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_name)
    return model, processor


if __name__ == "__main__":
    app()
