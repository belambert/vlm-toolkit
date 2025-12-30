from io import BytesIO

import torch
import typer
from diffusers import DiffusionPipeline

from imggen.gcs import upload_image_to_gcs
from imggen.util import get_device

app = typer.Typer()


@app.command()
def main(
    prompt: str = typer.Argument(..., help="The prompt for image generation"),
    model: str = typer.Option(
        "black-forest-labs/FLUX.2-dev", help="Model to use for image generation"
    ),
    num_inference_steps: int = typer.Option(25, help="Number of denoising steps"),
):

    pipe = DiffusionPipeline.from_pretrained(
        model,
        torch_dtype=torch.bfloat16,
    )
    pipe.to(get_device())

    image = pipe(prompt, num_inference_steps=num_inference_steps).images[0]

    buffer = BytesIO()
    image.save(buffer, format="PNG")

    # Upload to GCS
    result = upload_image_to_gcs(buffer, bucket_name="imggen", prefix="new")

    print(f"Date folder: {result['date_folder']}")
    print(f"Filename: {result['filename']}")
    print(f"File uploaded to GCS: {result['public_url']}")
    print(f"Blob path: {result['blob_path']}")


if __name__ == "__main__":
    app()
