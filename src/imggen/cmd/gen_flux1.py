"""Generate images using FLUX.1 models."""

from pathlib import Path

import torch
import typer
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, FluxTransformer2DModel
from PIL import Image
from transformers import AutoTokenizer, CLIPTextModel, T5EncoderModel

from imggen.util import get_device

app = typer.Typer()


def encode_prompt(text_encoder, text_encoder_2, tokenizer, tokenizer_2, prompt, dev):
    """Encode text prompt with CLIP and T5."""
    # CLIP encoding
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt",
    )
    prompt_embeds = text_encoder(
        text_inputs.input_ids.to(dev), output_hidden_states=False
    )
    pooled_prompt_embeds = prompt_embeds.pooler_output

    # T5 encoding
    text_inputs_2 = tokenizer_2(
        prompt,
        padding="max_length",
        max_length=512,
        truncation=True,
        return_tensors="pt",
    )
    prompt_embeds_2 = text_encoder_2(text_inputs_2.input_ids.to(dev))[0]

    return prompt_embeds_2, pooled_prompt_embeds


@app.command()
def main(
    prompt: str = typer.Argument(..., help="Text prompt for image generation"),
    output: Path = typer.Option("output.png", help="Output image path"),
    model: str = typer.Option(
        "black-forest-labs/FLUX.1-dev", help="Model to use for generation"
    ),
    num_inference_steps: int = typer.Option(20, help="Number of denoising steps"),
    guidance_scale: float = typer.Option(3.5, help="Guidance scale for CFG"),
    height: int = typer.Option(1024, help="Image height"),
    width: int = typer.Option(1024, help="Image width"),
    seed: int = typer.Option(None, help="Random seed for reproducibility"),
):
    """Generate an image using FLUX.1.

    Suggested models:
    - black-forest-labs/FLUX.1-dev
    - black-forest-labs/FLUX.1-schnell
    """
    dev = get_device()
    dtype = torch.bfloat16 if dev == "cuda" else torch.float32
    print(f"Using device: {dev}, dtype: {dtype}")

    # Set seed if provided
    if seed is not None:
        torch.manual_seed(seed)
        print(f"Using seed: {seed}")

    # Load models
    print(f"Loading models from {model}...")

    tokenizer = AutoTokenizer.from_pretrained(model, subfolder="tokenizer")
    tokenizer_2 = AutoTokenizer.from_pretrained(model, subfolder="tokenizer_2")

    print("loading encoder1")
    text_encoder = CLIPTextModel.from_pretrained(
        model, subfolder="text_encoder", torch_dtype=dtype
    ).to(dev)
    print("loading encoder2")
    text_encoder_2 = T5EncoderModel.from_pretrained(
        model, subfolder="text_encoder_2", torch_dtype=dtype
    ).to(dev)
    print("loading vae")
    vae = AutoencoderKL.from_pretrained(
        model, subfolder="vae", torch_dtype=dtype
    ).to(dev)

    print("loading vision transformer")
    # transformer = FluxTransformer2DModel.from_pretrained(
    #     model, subfolder="transformer", torch_dtype=dtype
    # ).to(dev)

    print("loading scheduler...")
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        model, subfolder="scheduler"
    )

    # Encode prompt
    print(f"Encoding prompt: {prompt}")
    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds = encode_prompt(
            text_encoder, text_encoder_2, tokenizer, tokenizer_2, prompt, dev
        )
        print(prompt_embeds.shape)
    return

    # Prepare latents
    latent_height = height // 8
    latent_width = width // 8
    latent_channels = 16

    # Pack latents (H, W, C) -> (H*W/4, C*4)
    packed_latent_height = latent_height // 2
    packed_latent_width = latent_width // 2
    packed_latent_channels = latent_channels * 4

    latents = torch.randn(
        1,
        packed_latent_height * packed_latent_width,
        packed_latent_channels,
        device=dev,
        dtype=dtype,
    )

    # Prepare position IDs
    latent_image_ids = torch.zeros(
        packed_latent_height, packed_latent_width, 3, device=dev, dtype=dtype
    )
    latent_image_ids[..., 1] = (
        latent_image_ids[..., 1]
        + torch.arange(packed_latent_height, device=dev, dtype=dtype)[:, None]
    )
    latent_image_ids[..., 2] = (
        latent_image_ids[..., 2]
        + torch.arange(packed_latent_width, device=dev, dtype=dtype)[None, :]
    )
    latent_image_ids = latent_image_ids.reshape(-1, 3)

    txt_ids = torch.zeros(prompt_embeds.shape[1], 3, device=dev, dtype=dtype)

    guidance = torch.full((1,), guidance_scale, device=dev, dtype=dtype)

    # Set timesteps
    scheduler.set_timesteps(num_inference_steps)

    # Denoising loop
    print("Generating image...")
    with torch.no_grad():
        for i, t in enumerate(scheduler.timesteps):
            timestep = t.unsqueeze(0).to(dev, dtype=dtype)

            # Predict noise
            noise_pred = transformer(
                hidden_states=latents,
                timestep=timestep,
                guidance=guidance,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt_embeds,
                img_ids=latent_image_ids,
                txt_ids=txt_ids,
                return_dict=False,
            )[0]

            # Scheduler step
            latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

    # Unpack latents (H*W/4, C*4) -> (H, W, C)
    latents = latents.reshape(
        1, packed_latent_height, packed_latent_width, latent_channels, 4
    )
    latents = latents.permute(0, 3, 1, 4, 2, 5)
    latents = latents.reshape(1, latent_channels, latent_height, latent_width)

    # Decode latents
    print("Decoding latents...")
    latents = (latents / vae.config.scaling_factor) + vae.config.shift_factor
    image = vae.decode(latents, return_dict=False)[0]

    # Convert to PIL
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).float().numpy()[0]
    image = Image.fromarray((image * 255).round().astype("uint8"))

    # Save image
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    print(f"Image saved to: {output}")


if __name__ == "__main__":
    app()
