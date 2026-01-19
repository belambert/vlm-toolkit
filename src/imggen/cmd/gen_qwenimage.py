"""Generate images using QwenImage model."""

from pathlib import Path

import torch
import typer
from diffusers import (
    AutoencoderKLQwenImage,
    FlowMatchEulerDiscreteScheduler,
    QwenImageTransformer2DModel,
)
from PIL import Image
from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration

from imggen.util import get_device

app = typer.Typer()

PROMPT_TEMPLATE = """<|im_start|>system
Describe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>
<|im_start|>user
{}<|im_end|>
<|im_start|>assistant
"""


def encode_prompt(text_encoder, tokenizer, prompt, dev, max_length=1024):
    """Encode text prompt with Qwen text encoder."""
    # Apply prompt template
    formatted_prompt = PROMPT_TEMPLATE.format(prompt)

    # Tokenize
    text_inputs = tokenizer(
        formatted_prompt,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    )

    # Encode
    with torch.no_grad():
        outputs = text_encoder(
            input_ids=text_inputs.input_ids.to(dev),
            attention_mask=text_inputs.attention_mask.to(dev),
            output_hidden_states=True,
        )
        prompt_embeds = outputs.hidden_states[-1]

    # Get actual sequence length (sum of attention mask)
    seq_len = text_inputs.attention_mask.sum().item()

    # Trim embeddings to actual sequence length
    prompt_embeds = prompt_embeds[:, :seq_len, :]

    return prompt_embeds, seq_len


@app.command()
def main(
    prompt: str = typer.Argument(..., help="Text prompt for image generation"),
    output: Path = typer.Option("output.png", help="Output image path"),
    model: str = typer.Option("Qwen/Qwen-Image", help="Model to use for generation"),
    num_inference_steps: int = typer.Option(28, help="Number of denoising steps"),
    height: int = typer.Option(1024, help="Image height"),
    width: int = typer.Option(1024, help="Image width"),
    seed: int = typer.Option(None, help="Random seed for reproducibility"),
):
    """Generate an image using QwenImage.

    Suggested models:
    - Qwen/Qwen-Image (main model)
    - Qwen/Qwen-Image-2512 (December 2025 version)
    - Qwen/Qwen-Image-Edit (for image editing)
    """
    dev = get_device()
    # dtype = torch.bfloat16 if dev == "cuda" else torch.float32
    dtype = torch.bfloat16
    print(f"Using device: {dev}, dtype: {dtype}")

    # Set seed if provided
    if seed is not None:
        torch.manual_seed(seed)
        print(f"Using seed: {seed}")

    # Load models
    print(f"Loading models from {model}...")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model, subfolder="tokenizer")

    print("Loading text encoder...")
    text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model, subfolder="text_encoder", dtype=dtype
    )
    text_encoder = text_encoder.to(dev)
    text_encoder.eval()

    print("Loading VAE...")
    vae = AutoencoderKLQwenImage.from_pretrained(
        model, subfolder="vae"
    )  # , dtype=dtype)
    vae = vae.to(dev)
    vae.eval()

    print("Loading scheduler...")
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        model, subfolder="scheduler"
    )

    # Encode prompt
    print(f"Encoding prompt: {prompt}")
    prompt_embeds, prompt_seq_len = encode_prompt(text_encoder, tokenizer, prompt, dev)

    print("text input shape")
    print(prompt_embeds.shape)

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

    # Set timesteps with mu for dynamic shifting
    mu = (packed_latent_height * packed_latent_width) / (128 * 128)
    scheduler.set_timesteps(num_inference_steps, mu=mu)

    print("removing encoder from memory...")
    del text_encoder

    print("moving VAE to CPU to save memory...")
    vae = vae.cpu()

    print("ensuring prompt embeds on device...")
    prompt_embeds = prompt_embeds.to(dev)

    print("Loading transformer...")
    transformer = QwenImageTransformer2DModel.from_pretrained(
        model, subfolder="transformer", torch_dtype=dtype
    )
    transformer = transformer.to(dev)
    transformer.eval()

    # Denoising loop
    print("Generating image...")

    with torch.no_grad():
        for i, t in enumerate(scheduler.timesteps):
            print(i)
            timestep = t.unsqueeze(0).to(dev, dtype=dtype)

            # Predict noise
            img_shapes = [(1, packed_latent_height, packed_latent_width)]
            txt_seq_lens = [prompt_seq_len]

            noise_pred = transformer(
                hidden_states=latents,
                timestep=timestep,
                encoder_hidden_states=prompt_embeds,
                img_shapes=img_shapes,
                txt_seq_lens=txt_seq_lens,
                return_dict=True,
            ).sample

            # Scheduler step
            latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

            if (i + 1) % 5 == 0:
                print(f"Step {i + 1}/{num_inference_steps}")

    # Unpack latents (H*W/4, C*4) -> (H, W, C)
    print("Unpacking latents...")
    latents = latents.reshape(
        1, packed_latent_height, packed_latent_width, latent_channels, 2, 2
    )
    latents = latents.permute(0, 3, 1, 4, 2, 5)
    latents = latents.reshape(1, latent_channels, latent_height, latent_width)

    print("removing transformer from memory...")
    del transformer

    print("moving VAE back to device for decoding...")
    vae = vae.to(dev)

    # Decode latents
    print("Decoding latents...")
    scaling_factor = vae.config.get("scaling_factor", 0.18215)
    latents = latents / scaling_factor

    # Add frame dimension for video VAE (B, C, H, W) -> (B, C, F, H, W)
    latents = latents.unsqueeze(2)

    # Convert to float32 for VAE decoding (MPS bfloat16 compatibility issue)
    latents = latents.float()

    image = vae.decode(latents, return_dict=False)[0]

    # Remove frame dimension (B, C, F, H, W) -> (B, C, H, W)
    image = image.squeeze(2)

    # Convert to PIL
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).float().detach().numpy()[0]
    image = Image.fromarray((image * 255).round().astype("uint8"))

    # Save image
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    print(f"Image saved to: {output}")


if __name__ == "__main__":
    app()
