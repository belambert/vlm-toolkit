import json
from pathlib import Path

import torch
import typer
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

app = typer.Typer()


def load_model(model_name: str, device: str):
    """Load Qwen2-VL model and processor."""
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if device != "cpu" else torch.float32,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_name)
    return model, processor


def detect_watermark(model, processor, image_path: Path, device: str) -> dict:
    """Detect watermark in an image using Qwen2-VL."""
    # Load and resize image if needed
    image = Image.open(image_path)
    width, height = image.size

    # Scale down if larger than 1024x1024
    max_size = 1024
    if width > max_size or height > max_size:
        # Calculate new size maintaining aspect ratio
        if width > height:
            new_width = max_size
            new_height = int(height * (max_size / width))
        else:
            new_height = max_size
            new_width = int(width * (max_size / height))

        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    prompt = """Look at this image carefully. Does it contain any watermarks, logos, text overlays, or copyright marks?

If yes, describe:
1. What the watermark says or looks like
2. Where it is located (e.g., top-left corner, bottom-right, center, etc.)
3. How prominent/visible it is

If no watermark is present, just say "No watermark detected."

Be specific about the location using terms like: top-left, top-center, top-right, middle-left, center, middle-right, bottom-left, bottom-center, bottom-right."""

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    # Prepare inputs
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(device)

    # Generate detection
    generated_ids = model.generate(**inputs, max_new_tokens=512, repetition_penalty=1.2)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    response = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    return {
        "image_path": str(image_path),
        "original_size": {"width": width, "height": height},
        "detection": response,
    }


@app.command()
def main(
    folder: Path = typer.Argument(..., help="Folder containing images to check"),
    output: Path = typer.Option(
        None,
        help="Output JSON file (default: watermark_detections.json in input folder)",
    ),
    extensions: str = typer.Option(
        "jpg,jpeg,png,webp", help="Comma-separated list of image extensions to process"
    ),
    model_name: str = typer.Option(
        "Qwen/Qwen2-VL-7B-Instruct", help="Hugging Face model name"
    ),
):
    """Detect watermarks in images using Qwen2-VL."""

    if not folder.exists() or not folder.is_dir():
        typer.echo(f"Error: {folder} is not a valid directory")
        raise typer.Exit(1)

    # Parse extensions
    ext_list = [f".{ext.strip().lower().lstrip('.')}" for ext in extensions.split(",")]

    # Find all images
    image_files = []
    for ext in ext_list:
        image_files.extend(folder.glob(f"*{ext}"))

    if not image_files:
        typer.echo(f"No images found in {folder} with extensions: {ext_list}")
        raise typer.Exit(1)

    typer.echo(f"Found {len(image_files)} images to check")

    # Determine device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    typer.echo(f"Using device: {device}")
    typer.echo(f"Loading model: {model_name}...")

    # Load model
    model, processor = load_model(model_name, device)

    typer.echo("Model loaded successfully\n")

    results = []

    for image_path in image_files:
        typer.echo(f"Checking: {image_path.name}...", nl=False)
        try:
            result = detect_watermark(model, processor, image_path, device)
            results.append(result)

            # Show brief result
            print(result)
            detection = result["detection"]
            if "no watermark" in detection.lower():
                typer.echo(" ✓ No watermark")
            else:
                typer.echo(" ⚠ Watermark detected")

        except Exception as e:
            typer.echo(f" ✗ Error: {e}")
            continue

    # Save results to JSON
    if output is None:
        output = folder / "watermark_detections.json"

    with open(output, "w") as f:
        json.dump(results, f, indent=2)

    typer.echo(f"\nResults saved to: {output}")
    typer.echo(f"Processed: {len(results)}/{len(image_files)} images")


if __name__ == "__main__":
    app()
