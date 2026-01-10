"""Print FLUX.1 model architecture."""

import torch
import typer
from diffusers import FluxTransformer2DModel

app = typer.Typer()


@app.command()
def main(
    model: str = typer.Option("black-forest-labs/FLUX.1-dev", help="Model to load"),
):
    """Load FLUX model and print architecture."""
    print(f"Loading {model}...")
    transformer = FluxTransformer2DModel.from_pretrained(
        model, subfolder="transformer", torch_dtype=torch.bfloat16
    )
    print("\n" + "=" * 80)
    print("FLUX Transformer Architecture")
    print("=" * 80 + "\n")
    print(transformer)


if __name__ == "__main__":
    app()
