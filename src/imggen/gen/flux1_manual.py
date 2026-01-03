import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.models import AutoencoderKL
from diffusers.models.transformers import FluxTransformer2DModel
from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast

# Generation parameters
MODEL = "black-forest-labs/FLUX.1-dev"
PROMPT = "a photo of an astronaut riding a horse on mars"
HEIGHT = 512
WIDTH = 512
NUM_INFERENCE_STEPS = 25
GUIDANCE_SCALE = 3.5  # FLUX uses lower guidance than SD
SEED = 42


# Helper function to pack latents for FLUX (2x2 patches)
def pack_latents(latents):
    # latents: (B, C, H, W) -> (B, H//2 * W//2, C*4)
    b, c, h, w = latents.shape
    latents = latents.view(b, c, h // 2, 2, w // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    latents = latents.reshape(b, (h // 2) * (w // 2), c * 4)
    return latents


def unpack_latents(latents, height, width):
    # latents: (B, H//2 * W//2, C*4) -> (B, C, H, W)
    b, seq_len, channels = latents.shape
    c = channels // 4
    h, w = height, width
    latents = latents.view(b, h // 2, w // 2, c, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)
    latents = latents.reshape(b, c, h, w)
    return latents


# Prepare position IDs for FLUX transformer
def prepare_latent_image_ids(height, width, device, dtype):
    # Create position IDs for image patches (after 2x2 packing)
    latent_h, latent_w = height // 2, width // 2
    img_ids = torch.zeros(latent_h, latent_w, 3, device=device, dtype=dtype)
    img_ids[..., 1] = (
        img_ids[..., 1] + torch.arange(latent_h, device=device, dtype=dtype)[:, None]
    )
    img_ids[..., 2] = (
        img_ids[..., 2] + torch.arange(latent_w, device=device, dtype=dtype)[None, :]
    )
    img_ids = img_ids.reshape(-1, 3)
    return img_ids


def prepare_text_ids(text_len, device, dtype):
    # Create position IDs for text tokens
    txt_ids = torch.zeros(text_len, 3, device=device, dtype=dtype)
    return txt_ids


def main():
    # Device
    device = "cuda" if torch.cuda.is_available() else "mps"
    dtype = torch.bfloat16  # FLUX uses bfloat16

    # 1. Load CLIP text encoder (for prompt embeddings)
    print("Loading CLIP...")
    clip_tok = CLIPTokenizer.from_pretrained(MODEL, subfolder="tokenizer")
    clip_enc = CLIPTextModel.from_pretrained(
        MODEL, subfolder="text_encoder", dtype=dtype
    )
    clip_enc = clip_enc.to(device)

    # 2. Load T5 text encoder (for additional conditioning)
    print("Loading T5...")
    t5_tok = T5TokenizerFast.from_pretrained(MODEL, subfolder="tokenizer_2")
    t5_enc = T5EncoderModel.from_pretrained(
        MODEL, subfolder="text_encoder_2", dtype=dtype
    )
    t5_enc = t5_enc.to(device)

    # 3. Load FLUX transformer (the main diffusion model)
    print("Loading FLUX transformer...")
    transf = FluxTransformer2DModel.from_pretrained(
        MODEL, subfolder="transformer", torch_dtype=dtype
    )
    transf = transf.to(device)

    # 4. Load VAE
    print("Loading VAE...")
    vae = AutoencoderKL.from_pretrained(MODEL, subfolder="vae", torch_dtype=dtype)
    vae = vae.to(device)

    # 5. Load scheduler
    print("Loading scheduler...")
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        MODEL, subfolder="scheduler"
    )

    print("All models loaded!")

    # Set random seed
    generator = torch.Generator(device=device).manual_seed(SEED)

    # 1. Encode prompt with CLIP
    print("Encoding prompt with CLIP...")
    clip_inputs = clip_tok(
        PROMPT,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt",
    )

    with torch.no_grad():
        clip_prompt_embeds = clip_enc(
            clip_inputs.input_ids.to(device), output_hidden_states=True
        )
        # FLUX uses pooled output from CLIP
        pooled_prompt_embeds = clip_prompt_embeds.pooler_output

    # 2. Encode prompt with T5
    print("Encoding prompt with T5...")
    t5_inputs = t5_tok(
        PROMPT,
        padding="max_length",
        max_length=512,
        truncation=True,
        return_tensors="pt",
    )

    with torch.no_grad():
        t5_prompt_embeds = t5_enc(t5_inputs.input_ids.to(device))[0]

    # 3. Prepare latents
    print("Preparing latents...")
    # FLUX VAE produces 16-channel latents
    latent_channels = 16
    latent_height = HEIGHT // 8
    latent_width = WIDTH // 8

    latents = torch.randn(
        (1, latent_channels, latent_height, latent_width),
        generator=generator,
        device=device,
        dtype=dtype,
    )

    # 4. Set timesteps
    scheduler.set_timesteps(NUM_INFERENCE_STEPS, device=device, mu=1.0)
    timesteps = scheduler.timesteps

    # Prepare position IDs (these stay constant throughout the denoising)
    img_ids = prepare_latent_image_ids(latent_height, latent_width, device, dtype)
    txt_ids = prepare_text_ids(t5_prompt_embeds.shape[1], device, dtype)

    intermediate_latents = []
    # 5. Denoising loop
    print("Starting denoising loop...")
    for i, t in enumerate(timesteps):
        print(f"Step {i+1}/{NUM_INFERENCE_STEPS}")

        # Pack latents for FLUX transformer
        latents_packed = pack_latents(latents)

        # FLUX uses guidance as a model input, not CFG
        timestep = t.expand(latents.shape[0])
        guidance = torch.tensor([GUIDANCE_SCALE], device=device, dtype=dtype).expand(
            latents.shape[0]
        )

        # Predict noise
        with torch.no_grad():
            noise_pred = transf(
                hidden_states=latents_packed,
                timestep=timestep / 1000,  # FLUX normalizes timesteps to [0, 1]
                guidance=guidance,
                encoder_hidden_states=t5_prompt_embeds,
                pooled_projections=pooled_prompt_embeds,
                txt_ids=txt_ids,
                img_ids=img_ids,
                return_dict=False,
            )[0]

        # Unpack the predicted noise
        noise_pred = unpack_latents(noise_pred, latent_height, latent_width)

        # Compute previous sample
        latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        intermediate_latents.append(latents)

    # 6. Decode latents to image
    print("Decoding latents...")
    # # FLUX VAE scaling factor
    # latents = latents / vae.config.scaling_factor

    # with torch.no_grad():
    #     image = vae.decode(latents, return_dict=False)[0]

    # 7. Convert to PIL Image
    # image = (image / 2 + 0.5).clamp(0, 1)
    # image = image.cpu().permute(0, 2, 3, 1).float().numpy()
    # image = (image * 255).round().astype("uint8")

    # pil_image = Image.fromarray(image[0])

    imgs = [latents_to_img(vae, latent) for latent in intermediate_latents]

    imgs[0].save(
        "animation.gif",
        save_all=True,
        append_images=imgs[1:],
        duration=200,  # milliseconds per frame
        loop=3,  # 0 means loop forever
    )

    pil_image = latents_to_img(vae, latents)
    # pil_image = tensor_to_img(image)
    pil_image.save("flux_output.png")
    print("Image saved as flux_output.png")


def latents_to_img(vae, latents: torch.Tensor):
    latents = latents / vae.config.scaling_factor
    with torch.no_grad():
        image = vae.decode(latents, return_dict=False)[0]
    return tensor_to_img(image)


def tensor_to_img(image: torch.Tensor):
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).float().numpy()
    image = (image * 255).round().astype("uint8")
    pil_image = Image.fromarray(image[0])
    return pil_image


if __name__ == "__main__":
    main()
