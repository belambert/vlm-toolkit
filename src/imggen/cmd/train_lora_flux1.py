"""Train a FLUX.1 LoRA adapter on images with captions."""

import json
from pathlib import Path

import torch
import torch.nn.functional as F
import typer
import yaml
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    FluxPipeline,
    FluxTransformer2DModel,
)
from diffusers.optimization import get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, CLIPTextModel, T5EncoderModel

import wandb

# Training hyperparameters
IMAGE_SIZE = 1024
BATCH_SIZE = 1
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
LR_WARMUP_STEPS = 500
SAVE_IMAGE_EPOCHS = 2
SAVE_MODEL_EPOCHS = 50
GRADIENT_ACCUMULATION_STEPS = 4  # Simulate larger batch size

# LoRA configuration
LORA_RANK = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0.0

# Model names
MODEL_NAME = "black-forest-labs/FLUX.1-dev"

# Validation prompts
VALIDATION_PROMPTS = [
    "a beautiful landscape",
    "a portrait of a person",
    "an abstract painting",
]

app = typer.Typer()


class ImageCaptionDataset(Dataset):
    """Dataset for images with text captions."""

    def __init__(self, caption_file: Path, transform=None):
        self.transform = transform
        self.images = []
        self.captions = []

        # Get caption file directory for resolving relative paths
        caption_dir = caption_file.parent

        # Load from JSONL file
        with open(caption_file) as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    img_path = Path(data["image_path"])
                    # Resolve relative to caption file directory
                    if not img_path.is_absolute():
                        img_path = caption_dir / img_path
                    if img_path.exists():
                        self.images.append(img_path)
                        self.captions.append(data["output"])

        if len(self.images) == 0:
            raise ValueError(f"No valid images found in {caption_file}")

        print(f"Loaded {len(self.images)} images from {caption_file}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return {"images": image, "captions": self.captions[idx]}


def encode_prompt(
    text_encoder, text_encoder_2, tokenizer, tokenizer_2, prompt, dev, dtype
):
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
    pooled_prompt_embeds = prompt_embeds.pooler_output.to(dtype)

    # T5 encoding
    text_inputs_2 = tokenizer_2(
        prompt,
        padding="max_length",
        max_length=512,
        truncation=True,
        return_tensors="pt",
    )
    prompt_embeds_2 = text_encoder_2(text_inputs_2.input_ids.to(dev))[0].to(dtype)

    return prompt_embeds_2, pooled_prompt_embeds


def generate_validation_images(
    transformer,
    vae,
    text_encoder,
    text_encoder_2,
    tokenizer,
    tokenizer_2,
    scheduler,
    dev,
    dtype,
    lora_config,
    learning_rate,
    global_step,
    validation_prompts,
):
    """Generate validation images and return updated transformer and optimizer."""
    transformer.eval()

    # Merge LoRA weights for inference
    transformer = transformer.merge_and_unload()

    pipeline = FluxPipeline(
        vae=vae,
        text_encoder=text_encoder,
        text_encoder_2=text_encoder_2,
        tokenizer=tokenizer,
        tokenizer_2=tokenizer_2,
        transformer=transformer,
        scheduler=scheduler,
    )
    pipeline = pipeline.to(dev)

    with torch.no_grad():
        for i, prompt in enumerate(validation_prompts):
            images = pipeline(
                prompt,
                num_inference_steps=20,
                guidance_scale=3.5,
                height=IMAGE_SIZE,
                width=IMAGE_SIZE,
            ).images

            wandb.log(
                {f"validation_{i}": wandb.Image(images[0], caption=prompt)},
                step=global_step,
            )

    # Unmerge LoRA weights to continue training
    transformer = get_peft_model(
        FluxTransformer2DModel.from_pretrained(
            MODEL_NAME, subfolder="transformer", torch_dtype=dtype
        ).to(dev),
        lora_config,
    )
    # Reload optimizer state
    optimizer = torch.optim.AdamW(transformer.parameters(), lr=learning_rate)

    return transformer, optimizer


@app.command()
def main(
    caption_file: Path = typer.Argument(
        ..., help="JSONL file with image paths and captions"
    ),
    output_dir: Path = typer.Option(
        "flux1-lora", help="Output directory for LoRA checkpoints"
    ),
    validation_prompts_file: Path = typer.Option(
        None, help="YAML file with validation prompts"
    ),
    num_epochs: int = typer.Option(NUM_EPOCHS, help="Number of training epochs"),
    batch_size: int = typer.Option(BATCH_SIZE, help="Batch size for training"),
    learning_rate: float = typer.Option(LEARNING_RATE, help="Learning rate"),
):
    """Train a FLUX.1 LoRA adapter.

    Expects a JSONL file with entries containing 'image_path' and 'output' (caption) fields.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Load validation prompts
    if validation_prompts_file is not None:
        with open(validation_prompts_file) as f:
            validation_prompts = yaml.safe_load(f)
    else:
        validation_prompts = VALIDATION_PROMPTS

    # Initialize wandb
    wandb.init(
        project="flux1-lora-training",
        config={
            "image_size": IMAGE_SIZE,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "learning_rate": learning_rate,
            "lora_rank": LORA_RANK,
            "lora_alpha": LORA_ALPHA,
        },
    )

    # Setup device
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        dtype = torch.bfloat16
    elif torch.backends.mps.is_available():
        dev = torch.device("mps")
        dtype = torch.float32  # MPS has issues with bfloat16
    else:
        dev = torch.device("cpu")
        dtype = torch.float32
    print(f"Using device: {dev}, dtype: {dtype}")

    # Load tokenizers and text encoders
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, subfolder="tokenizer")
    tokenizer_2 = AutoTokenizer.from_pretrained(MODEL_NAME, subfolder="tokenizer_2")

    text_encoder = CLIPTextModel.from_pretrained(
        MODEL_NAME, subfolder="text_encoder", torch_dtype=dtype
    ).to(dev)
    text_encoder_2 = T5EncoderModel.from_pretrained(
        MODEL_NAME, subfolder="text_encoder_2", torch_dtype=dtype
    ).to(dev)

    # Freeze text encoders
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    text_encoder.eval()
    text_encoder_2.eval()

    # Load VAE
    vae = AutoencoderKL.from_pretrained(
        MODEL_NAME, subfolder="vae", torch_dtype=dtype
    ).to(dev)
    vae.requires_grad_(False)
    vae.eval()

    # Load transformer and add LoRA layers
    transformer = FluxTransformer2DModel.from_pretrained(
        MODEL_NAME, subfolder="transformer", torch_dtype=dtype
    ).to(dev)

    # Enable gradient checkpointing for memory efficiency
    transformer.enable_gradient_checkpointing()

    # Configure LoRA
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        lora_dropout=LORA_DROPOUT,
    )
    transformer = get_peft_model(transformer, lora_config)
    transformer.print_trainable_parameters()

    # Create noise scheduler
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        MODEL_NAME, subfolder="scheduler"
    )

    # Setup optimizer
    optimizer = torch.optim.AdamW(transformer.parameters(), lr=learning_rate)

    # Prepare dataset
    transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    dataset = ImageCaptionDataset(caption_file, transform=transform)

    # Use multiple workers for CUDA, 0 for MPS
    num_workers = 0 if dev.type == "mps" else 4

    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    # Setup learning rate scheduler
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=LR_WARMUP_STEPS,
        num_training_steps=len(dataloader) * num_epochs,
    )

    print(f"Training on {len(dataset)} images")

    # Training loop
    global_step = 0
    for epoch in range(num_epochs):
        transformer.train()
        progress_bar = tqdm(total=len(dataloader), desc=f"Epoch {epoch}")

        for step, batch in enumerate(dataloader):
            images = batch["images"].to(dev, dtype=dtype)
            captions = batch["captions"]

            # Encode images to latent space
            with torch.no_grad():
                latents = vae.encode(images).latent_dist.sample()
                latents = (
                    latents - vae.config.shift_factor
                ) * vae.config.scaling_factor

                # Pack latents for FLUX transformer (B, C, H, W) -> (B, H*W/4, C*4)
                bs, c, h, w = latents.shape
                latents = latents.permute(0, 2, 3, 1).reshape(
                    bs, (h // 2) * (w // 2), c * 4
                )

            # Sample noise
            noise = torch.randn_like(latents)
            bs = latents.shape[0]

            # Sample random timesteps (FLUX uses continuous time)
            timesteps = torch.rand((bs,), device=dev, dtype=dtype)

            # Add noise using flow matching
            noisy_latents = (1 - timesteps.view(-1, 1, 1)) * latents + timesteps.view(
                -1, 1, 1
            ) * noise

            # Encode prompts
            with torch.no_grad():
                prompt_embeds, pooled_prompt_embeds = encode_prompt(
                    text_encoder,
                    text_encoder_2,
                    tokenizer,
                    tokenizer_2,
                    captions,
                    dev,
                    dtype,
                )

            # Create position IDs for rotary embeddings
            latent_image_ids = torch.zeros(h // 2, w // 2, 3, device=dev, dtype=dtype)
            latent_image_ids[..., 1] = (
                latent_image_ids[..., 1]
                + torch.arange(h // 2, device=dev, dtype=dtype)[:, None]
            )
            latent_image_ids[..., 2] = (
                latent_image_ids[..., 2]
                + torch.arange(w // 2, device=dev, dtype=dtype)[None, :]
            )
            latent_image_ids = latent_image_ids.reshape(-1, 3)

            txt_ids = torch.zeros(prompt_embeds.shape[1], 3, device=dev, dtype=dtype)

            # Create guidance embedding (for CFG, set to 3.5 which is a typical value)
            guidance = torch.full((bs,), 3.5, device=dev, dtype=dtype)

            # Predict velocity
            model_output = transformer(
                hidden_states=noisy_latents,
                timestep=timesteps,
                guidance=guidance,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt_embeds,
                img_ids=latent_image_ids,
                txt_ids=txt_ids,
                return_dict=False,
            )[0]

            # Calculate flow matching loss (velocity prediction)
            target = noise - latents
            loss = F.mse_loss(model_output, target)

            loss.backward()

            if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            # Log metrics
            logs = {"loss": loss.item(), "lr": lr_scheduler.get_last_lr()[0]}
            wandb.log(logs, step=global_step)
            progress_bar.set_postfix(logs)
            progress_bar.update(1)

        progress_bar.close()

        # Generate validation images
        if (epoch + 1) % SAVE_IMAGE_EPOCHS == 0:
            transformer, optimizer = generate_validation_images(
                transformer,
                vae,
                text_encoder,
                text_encoder_2,
                tokenizer,
                tokenizer_2,
                scheduler,
                dev,
                dtype,
                lora_config,
                learning_rate,
                global_step,
                validation_prompts,
            )

        # Save checkpoint
        if (epoch + 1) % SAVE_MODEL_EPOCHS == 0:
            checkpoint_dir = output_dir / f"checkpoint-{epoch + 1}"
            checkpoint_dir.mkdir(exist_ok=True)

            # Save only LoRA weights
            transformer_lora = transformer.merge_and_unload()
            transformer_lora.save_pretrained(checkpoint_dir)

            print(f"Saved checkpoint to {checkpoint_dir}")

    # Save final LoRA weights
    final_dir = output_dir / "final"
    final_dir.mkdir(exist_ok=True)
    transformer_final = transformer.merge_and_unload()
    transformer_final.save_pretrained(final_dir)
    print(f"Training complete. Final model saved to {final_dir}")

    wandb.finish()


if __name__ == "__main__":
    app()
