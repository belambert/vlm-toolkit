from io import BytesIO

import torch
import typer
from diffusers import FluxPipeline

# https://huggingface.co/docs/diffusers/main/en/api/pipelines/flux

# maybe this code?
# https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/flux/pipeline_flux.py

# also see:
# https://huggingface.co/blog/sd3#memory-optimizations-for-sd3

app = typer.Typer()


@app.command()
def main(
    prompt: str = typer.Argument(..., help="The prompt for image generation"),
    output: typer.FileBinaryWrite = typer.Argument(
        "output.png", help="Output filename (use '-' for stdout)"
    ),
):
    pipe = FluxPipeline.from_pretrained(
        # "black-forest-labs/FLUX.1-dev",
        # "HurdyThirty/FluxedUp",
        # "Heartsync/Flux-NSFW-uncensored",
        "SicariusSicariiStuff/flux.1dev-abliteratedv2",
        torch_dtype=torch.bfloat16,
    )
    # save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power
    pipe.enable_model_cpu_offload()

    image = pipe(
        prompt,
        height=512,
        width=512,
        guidance_scale=3.5,
        num_inference_steps=25,
        max_sequence_length=512,
        generator=torch.Generator("cpu").manual_seed(0),
    ).images[0]

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    output.write(buffer.getvalue())


if __name__ == "__main__":
    app()
