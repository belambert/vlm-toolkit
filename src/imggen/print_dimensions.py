from pathlib import Path

import typer
from PIL import Image

app = typer.Typer()


@app.command()
def main(
    folder: Path = typer.Argument(..., help="Folder containing images"),
    extensions: str = typer.Option(
        "jpg,jpeg,png,webp,gif", help="Comma-separated list of image extensions"
    ),
):
    """Print pixel dimensions for all images in a folder."""

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

    # Sort by filename
    image_files.sort()

    typer.echo(f"Found {len(image_files)} images\n")
    typer.echo(f"{'Filename':<50} {'Width':>6} x {'Height':<6}")
    typer.echo("-" * 70)

    for image_path in image_files:
        try:
            with Image.open(image_path) as img:
                width, height = img.size
                typer.echo(f"{image_path.name:<50} {width:>6} x {height:<6}")
        except Exception as e:
            typer.echo(f"{image_path.name:<50} Error: {e}")


if __name__ == "__main__":
    app()
