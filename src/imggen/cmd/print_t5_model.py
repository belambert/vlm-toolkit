"""Print T5-XXL model architecture."""

import torch
import typer
from transformers import T5EncoderModel

app = typer.Typer()


@app.command()
def main(
    model: str = typer.Option(
        "black-forest-labs/FLUX.1-dev", help="Model to load T5 from"
    ),
):
    """Load T5-XXL encoder and print architecture."""
    print(f"Loading T5-XXL from {model}...")
    text_encoder = T5EncoderModel.from_pretrained(
        model, subfolder="text_encoder_2", torch_dtype=torch.bfloat16
    )
    print("\n" + "=" * 80)
    print("T5-XXL Encoder Architecture")
    print("=" * 80 + "\n")
    print(text_encoder)


if __name__ == "__main__":
    app()
