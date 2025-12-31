import json
from pathlib import Path

import torch
import typer
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

app = typer.Typer()

DEFAULT_PROMPT = """
Write a detailed, but brief caption for this image. Focus on the appearance and
positions of the people in the picture, as well as their general setting (e.g. in
a park, in a kitchen).  Refer to women in the picture as girls
and men as guys. Describe any sexual acts using common slang terms like
doggy, cowgirl, missionary, blowjob, creampie, cum etc. Describe the state of dress of
any people in the image (e.g. topless, naked, clothed). Describe any clothing briefly
e.g. pink bra, tight jeans. It's not necessary to describe background props, like
paintings or light fixtures. Include brief descriptions of each persons appearance, e.g
a tall, thin girl, or a girl with short brown hair.

The caption should read like it's the instructions from a user to an image generation
model.  Here are some good example captions:

Example: brunette girl with small boobs riding reverse cowgirl on a bed. Her mouth is
open and there's cum everywhere

Example: brunette with long brown hair sitting on a blanket in a park with her knees up.
She has one boob out and no pants on. She's gently spreading her pussy for the camera to
see

Example: hot naked girl with straight brown hair squatting and blowing a guy wearing only a tanktop. She's touching
her pussy. they are in an art studio.
"""

def load_model(model_name: str, device: str):
    """Load Qwen2-VL model and processor."""
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if device != "cpu" else torch.float32,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_name)
    return model, processor


def caption_image(model, processor, image_path: Path, prompt: str, device: str) -> str:
    """Generate a caption for a single image using Qwen2-VL."""
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

    # Generate caption
    generated_ids = model.generate(**inputs, max_new_tokens=512, repetition_penalty=1.2)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    caption = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    return caption


@app.command()
def main(
    folder: Path = typer.Argument(..., help="Folder containing images to caption"),
    output_format: str = typer.Option(
        "txt", help="Output format: 'txt' (one file per image) or 'json' (single file)"
    ),
    prompt: str = typer.Option(
        DEFAULT_PROMPT,
        help="Prompt to use for captioning",
    ),
    extensions: str = typer.Option(
        "jpg,jpeg,png,webp", help="Comma-separated list of image extensions to process"
    ),
    model_name: str = typer.Option(
        "Qwen/Qwen2-VL-7B-Instruct", help="Hugging Face model name"
    ),
):
    """Generate captions for all images in a folder using Qwen2-VL."""

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

    typer.echo(f"Found {len(image_files)} images to caption")

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

    results = {}

    for image_path in image_files:
        typer.echo(f"Processing: {image_path.name}...", nl=False)
        try:
            caption = caption_image(model, processor, image_path, prompt, device)
            results[image_path.name] = caption

            if output_format == "txt":
                # Save as .txt file next to image
                caption_path = image_path.with_suffix(".txt")
                caption_path.write_text(caption)
                typer.echo(f" ✓ (saved to {caption_path.name})")
            else:
                typer.echo(" ✓")

        except Exception as e:
            typer.echo(f" ✗ Error: {e}")
            continue

    # If JSON format, save all captions to a single file
    if output_format == "json":
        output_path = folder / "captions.json"
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        typer.echo(f"\nSaved all captions to {output_path}")

    typer.echo(f"\nCompleted: {len(results)}/{len(image_files)} images captioned")


if __name__ == "__main__":
    app()
