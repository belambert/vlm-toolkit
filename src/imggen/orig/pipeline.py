"""
Run a diffusion pipeline directly.
"""

import torch
from diffusers import AutoencoderKL, UNet2DConditionModel, UniPCMultistepScheduler
from PIL import Image, ImageDraw
from transformers import CLIPTextModel, CLIPTokenizer

PROMPT = ["a bluejay in an oak tree"]
HEIGHT = 512  # default height of Stable Diffusion default: 512
WIDTH = 512  # default width of Stable Diffusion default: 512
STEPS = 20  # Number of denoising steps
GUIDANCE_SCALE = 7.5  # Scale for classifier-free guidance


def main():
    print("loading models...")
    vae = AutoencoderKL.from_pretrained(
        "CompVis/stable-diffusion-v1-4", subfolder="vae", use_safetensors=True
    )
    tokenizer = CLIPTokenizer.from_pretrained(
        "CompVis/stable-diffusion-v1-4", subfolder="tokenizer"
    )
    encoder = CLIPTextModel.from_pretrained(
        "CompVis/stable-diffusion-v1-4", subfolder="text_encoder", use_safetensors=True
    )
    unet = UNet2DConditionModel.from_pretrained(
        "CompVis/stable-diffusion-v1-4", subfolder="unet", use_safetensors=True
    )
    scheduler = UniPCMultistepScheduler.from_pretrained(
        "CompVis/stable-diffusion-v1-4", subfolder="scheduler"
    )

    # device = get_device()
    device = "cpu"
    vae.to(device)
    encoder.to(device)
    unet.to(device)

    # Seed generator to create the initial latent noise
    generator = torch.manual_seed(0)
    batch_size = len(PROMPT)

    # encode the text
    tokens = tokenize(tokenizer, PROMPT)
    text_embeddings = encode(encoder, tokenizer, tokens, device)

    # initialize the latent image
    latents = torch.randn(
        (batch_size, unet.config.in_channels, HEIGHT // 8, WIDTH // 8),
        generator=generator,
        device=device,
    )
    latents = latents * scheduler.init_noise_sigma
    scheduler.set_timesteps(STEPS)
    filename = "gen"

    print("iterating...")
    for i, t in enumerate(scheduler.timesteps):
        print(i)
        # expand the latents if we are doing classifier-free guidance to avoid doing
        # two forward passes.
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)
        # predict the noise residual
        with torch.no_grad():
            noise_pred = unet(
                latent_model_input, t, encoder_hidden_states=text_embeddings
            ).sample
        # perform guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + GUIDANCE_SCALE * (
            noise_pred_text - noise_pred_uncond
        )
        # compute the previous noisy sample x_t -> x_t-1
        latents = scheduler.step(noise_pred, t, latents).prev_sample
        image = decode_latents(vae, latents)
        image.save(f"{filename}_{i}.png")
    # image = decode_latents(vae, latents)

    draw = ImageDraw.Draw(image)
    # font = ImageFont.truetype(<font-file>, <font-size>)
    # font = ImageFont.truetype("sans-serif.ttf", 16)
    # draw.text((x, y),"Sample Text",(r,g,b))
    draw.text((0, 0), PROMPT[0], (255, 255, 255))  # , font=font)
    filename = f"{filename}_final.png"
    print(f"saving image to: {filename}")
    image.save(filename)


def decode_latents(vae, latents: torch.Tensor):
    """scale and decode the image latents with vae"""
    print("decoding image...")
    latents = 1 / 0.18215 * latents
    with torch.no_grad():
        image: torch.Tensor = vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1).squeeze()
    image = (image.permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()
    # what's this one?
    # images = (image * 255).round().astype("uint8")
    # now it's a PIL Image
    return Image.fromarray(image)


def tokenize(tokenizer, prompt):
    print("tokenizing...")
    return tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )


def encode(encoder, tokenizer, tokens, device):
    print("encoding...")
    batch_size = 1
    with torch.no_grad():
        text_embeddings = encoder(tokens.input_ids.to(device))[0]

    max_length = tokens.input_ids.shape[-1]
    uncond_input = tokenizer(
        [""] * batch_size,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )
    uncond_embeddings = encoder(uncond_input.input_ids.to(device))[0]
    return torch.cat([uncond_embeddings, text_embeddings])


if __name__ == "__main__":
    main()
