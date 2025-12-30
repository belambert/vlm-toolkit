import uuid
from datetime import datetime
from io import BytesIO

import torch
import typer
from diffusers import DiffusionPipeline
from google.cloud import storage

app = typer.Typer()

from imggen.util import get_device


@app.command()
def main(
    prompt: str = typer.Argument(..., help="The prompt for image generation"),
    num_inference_steps: int = typer.Option(25, help="Number of denoising steps"),
):

    pipe = DiffusionPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        # "Tongyi-MAI/Z-Image-Turbo",
        torch_dtype=torch.bfloat16,
    )
    pipe.to(get_device())

    image = pipe(prompt, num_inference_steps=num_inference_steps).images[0]

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)

    # GCS configuration
    bucket_name = "imggen"
    prefix = "new"

    # Initialize GCS client
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    # Get current date for folder structure
    date_folder = datetime.now().strftime("%Y-%m-%d")

    # Generate unique filename using timestamp and UUID
    timestamp = datetime.now().strftime("%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    filename = f"image_{timestamp}_{unique_id}.png"

    # Construct full blob path: prefix/date/filename
    blob_path = f"{prefix}/{date_folder}/{filename}"
    blob = bucket.blob(blob_path)

    # Upload to GCS
    blob.upload_from_file(buffer, content_type="image/png")

    # Generate public URL
    public_url = f"gs://{bucket_name}/{blob_path}"

    print(f"Date folder: {date_folder}")
    print(f"Filename: {filename}")
    print(f"File uploaded to GCS: {public_url}")
    print(f"Blob path: {blob_path}")

if __name__ == "__main__":
    app()