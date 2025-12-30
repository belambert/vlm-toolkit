from io import BytesIO

import torch
import typer
from diffusers import AutoPipelineForText2Image
from peft import PeftConfig, PeftModel

from imggen.gcs import upload_image_to_gcs
from imggen.util import get_device

app = typer.Typer()


@app.command()
def main(
    prompt: str = typer.Argument(..., help="The prompt for image generation"),
):
    device = get_device()
    pipe = AutoPipelineForText2Image.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=torch.bfloat16,
    )
    pipe.to(device)

    pipe.load_lora_weights(
        "Heartsync/Flux-NSFW-uncensored",
        weight_name="lora.safetensors",
        adapter_name="uncensored",
    )

    negative_prompt = "text, watermark, signature, cartoon, anime, illustration, painting, drawing, low quality, blurry"
    seed = 42
    generator = torch.Generator(device=device).manual_seed(seed)

    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        guidance_scale=7.0,
        num_inference_steps=28,
        width=1024,
        height=1024,
        generator=generator,
    ).images[0]

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
