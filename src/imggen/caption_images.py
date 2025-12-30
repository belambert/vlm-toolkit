import base64
import json
from pathlib import Path

import anthropic
import typer

app = typer.Typer()


def encode_image(image_path: Path) -> tuple[str, str]:
    """Encode image to base64 and determine media type."""
    with open(image_path, "rb") as f:
        image_data = base64.standard_b64encode(f.read()).decode("utf-8")

    # Determine media type from extension
    extension = image_path.suffix.lower()
    media_type_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    media_type = media_type_map.get(extension, "image/jpeg")

    return image_data, media_type


def caption_image(client: anthropic.Anthropic, image_path: Path, prompt: str) -> str:
    """Generate a caption for a single image using Claude."""
    image_data, media_type = encode_image(image_path)

    message = client.messages.create(
        model="claude-3-5-sonnet-latest",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_data,
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    )

    return message.content[0].text


@app.command()
def main(
    folder: Path = typer.Argument(..., help="Folder containing images to caption"),
    output_format: str = typer.Option(
        "txt", help="Output format: 'txt' (one file per image) or 'json' (single file)"
    ),
    prompt: str = typer.Option(
        "Describe this image in detail. Be specific and descriptive.",
        help="Prompt to use for captioning",
    ),
    extensions: str = typer.Option(
        "jpg,jpeg,png,webp,gif",
        help="Comma-separated list of image extensions to process",
    ),
):
    """Generate captions for all images in a folder using Claude."""

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

    # Initialize Anthropic client
    client = anthropic.Anthropic()

    results = {}

    for image_path in image_files:
        typer.echo(f"Processing: {image_path.name}...", nl=False)
        try:
            caption = caption_image(client, image_path, prompt)
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
