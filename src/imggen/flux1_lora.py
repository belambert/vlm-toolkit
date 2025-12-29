from io import BytesIO

import torch
import typer
from diffusers import AutoPipelineForText2Image
from peft import PeftConfig, PeftModel

app = typer.Typer()

from imggen.util import get_device


@app.command()
def main(
    prompt: str = typer.Argument(..., help="The prompt for image generation"),
    output: typer.FileBinaryWrite = typer.Argument(
        "output.png", help="Output filename (use '-' for stdout)"
    ),
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

    prompt = "A woman in a sheer white dress standing on a beach at sunset, backlit so her silhouette is visible through the thin fabric, shot with Canon EOS R5, 85mm f/1.2 lens, golden hour natural lighting, professional composition, hyperrealistic detail, masterpiece quality, 8K resolution."
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
    output.write(buffer.getvalue())

if __name__ == "__main__":
    app()