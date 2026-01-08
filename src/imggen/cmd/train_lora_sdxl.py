"""Train an SDXL LoRA adapter on a folder of images with captions."""

from pathlib import Path

import torch
import torch.nn.functional as F
import typer
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, CLIPTextModel, CLIPTextModelWithProjection

import wandb

# Training hyperparameters
# IMAGE_SIZE = 1024
IMAGE_SIZE = 512
BATCH_SIZE = 1
NUM_EPOCHS = 100
LEARNING_RATE = 1e-4
LR_WARMUP_STEPS = 500
SAVE_IMAGE_EPOCHS = 1
SAVE_MODEL_EPOCHS = 50
GRADIENT_ACCUMULATION_STEPS = 1

# LoRA configuration
LORA_RANK = 4
LORA_ALPHA = 4
LORA_DROPOUT = 0.0

# Model names
MODEL_NAME = "stabilityai/stable-diffusion-xl-base-1.0"

# Validation prompts
VALIDATION_PROMPTS = [
    "sassafras albidum",
    "eryngium yuccifolium",
    "carya tomentosa",
]

app = typer.Typer()


class ImageCaptionDataset(Dataset):
    """Dataset for images with text captions."""

    def __init__(self, folder: Path, transform=None):
        self.folder = folder
        self.transform = transform

        # Find all image files
        extensions = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
        self.images = []
        for ext in extensions:
            self.images.extend(folder.glob(ext))

        if len(self.images) == 0:
            raise ValueError(
                f"No images found in {folder}. Looking for: {', '.join(extensions)}"
            )

        self.images = self.images[:100]
        print([x.name for x in self.images])

        # Load captions from txt files with same name as images
        self.captions = []
        for img_path in self.images:
            caption_path = img_path.with_suffix(".txt")
            if caption_path.exists():
                self.captions.append(caption_path.read_text().strip())
            else:
                # Use filename as caption if no txt file
                self.captions.append(img_path.stem)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return {"images": image, "captions": self.captions[idx]}


def encode_prompt(text_encoders, tokenizers, prompt, device):
    """Encode text prompt with both CLIP text encoders."""
    prompt_embeds_list = []

    for text_encoder, tokenizer in zip(text_encoders, tokenizers):
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(device)

        prompt_embeds = text_encoder(text_input_ids, output_hidden_states=True)

        # Use pooled output for text_encoder_2
        if len(prompt_embeds_list) == 1:
            pooled_prompt_embeds = prompt_embeds.text_embeds
        else:
            pooled_prompt_embeds = None

        # Use hidden states
        prompt_embeds = prompt_embeds.hidden_states[-2]
        prompt_embeds_list.append(prompt_embeds)

    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
    return prompt_embeds, pooled_prompt_embeds


def generate_validation_images(
    unet, vae, enc1, enc2, tok1, tok2, noise_scheduler, dev, lora_config, learning_rate, global_step,
):
    """Generate validation images and return updated unet and optimizer."""
    unet.eval()

    # Merge LoRA weights for inference
    unet = unet.merge_and_unload()

    pipeline = StableDiffusionXLPipeline(
        vae=vae,
        text_encoder=enc1,
        text_encoder_2=enc2,
        tokenizer=tok1,
        tokenizer_2=tok2,
        unet=unet,
        scheduler=noise_scheduler,
    )
    pipeline = pipeline.to(dev)

    with torch.no_grad():
        for i, prompt in enumerate(VALIDATION_PROMPTS):
            images = pipeline(
                prompt,
                num_inference_steps=30,
                guidance_scale=7.5,
            ).images

            wandb.log(
                {f"validation_{i}": wandb.Image(images[0], caption=prompt)},
                step=global_step,
            )

    # Unmerge LoRA weights to continue training
    unet = get_peft_model(
        UNet2DConditionModel.from_pretrained(MODEL_NAME, subfolder="unet").to(dev),
        lora_config,
    )
    # Reload optimizer state
    optimizer = torch.optim.AdamW(unet.parameters(), lr=learning_rate)

    return unet, optimizer


@app.command()
def main(
    data_dir: Path = typer.Argument(..., help="Folder containing training images"),
    output_dir: Path = typer.Option(
        "sdxl-lora", help="Output directory for LoRA checkpoints"
    ),
    num_epochs: int = typer.Option(NUM_EPOCHS, help="Number of training epochs"),
    batch_size: int = typer.Option(BATCH_SIZE, help="Batch size for training"),
    learning_rate: float = typer.Option(LEARNING_RATE, help="Learning rate"),
):
    """Train an SDXL LoRA adapter.

    Expects images with corresponding .txt files containing captions.
    If no .txt file exists, the filename will be used as the caption.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Initialize wandb
    wandb.init(
        project="sdxl-lora-training",
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
    elif torch.backends.mps.is_available():
        dev = torch.device("mps")
    else:
        dev = torch.device("cpu")
    print(f"Using device: {dev}")

    # Load tokenizers and text encoders
    tok1 = AutoTokenizer.from_pretrained(
        MODEL_NAME, subfolder="tokenizer", use_fast=False
    )
    tok2 = AutoTokenizer.from_pretrained(
        MODEL_NAME, subfolder="tokenizer_2", use_fast=False
    )

    enc1 = CLIPTextModel.from_pretrained(MODEL_NAME, subfolder="text_encoder").to(dev)
    enc2 = CLIPTextModelWithProjection.from_pretrained(
        MODEL_NAME, subfolder="text_encoder_2"
    ).to(dev)

    # Freeze text encoders
    enc1.requires_grad_(False)
    enc2.requires_grad_(False)

    # Load VAE
    vae = AutoencoderKL.from_pretrained(MODEL_NAME, subfolder="vae").to(dev)
    vae.requires_grad_(False)

    # Load UNet and add LoRA layers
    unet = UNet2DConditionModel.from_pretrained(MODEL_NAME, subfolder="unet").to(dev)

    # Configure LoRA
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        lora_dropout=LORA_DROPOUT,
    )
    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()

    # Create noise scheduler
    noise_scheduler = DDPMScheduler.from_pretrained(MODEL_NAME, subfolder="scheduler")

    # Setup optimizer
    optimizer = torch.optim.AdamW(unet.parameters(), lr=learning_rate)

    # Prepare dataset
    transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    dataset = ImageCaptionDataset(data_dir, transform=transform)

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
        unet.train()
        progress_bar = tqdm(total=len(dataloader), desc=f"Epoch {epoch}")

        for step, batch in enumerate(dataloader):
            images = batch["images"].to(dev)
            captions = batch["captions"]

            # Encode images to latent space
            with torch.no_grad():
                latents = vae.encode(images).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

            # Sample noise
            noise = torch.randn_like(latents)
            bs = latents.shape[0]

            # Sample random timesteps
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bs,), device=dev
            ).long()

            # Add noise to latents
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # Encode prompts
            with torch.no_grad():
                prompt_embeds, pooled_prompt_embeds = encode_prompt(
                    [enc1, enc2],
                    [tok1, tok2],
                    captions,
                    dev,
                )

            # Add time embeddings
            add_time_ids = (
                torch.tensor([[IMAGE_SIZE, IMAGE_SIZE, 0, 0, IMAGE_SIZE, IMAGE_SIZE]])
                .repeat(bs, 1)
                .to(dev)
            )

            # Prepare added conditioning
            added_cond_kwargs = {
                "text_embeds": pooled_prompt_embeds,
                "time_ids": add_time_ids,
            }

            # Predict noise
            model_output = unet(
                noisy_latents,
                timesteps,
                prompt_embeds,
                added_cond_kwargs=added_cond_kwargs,
            ).sample

            # Calculate loss
            loss = F.mse_loss(model_output, noise)

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
            unet, optimizer = generate_validation_images(
                unet, vae, enc1, enc2, tok1, tok2, noise_scheduler, dev, lora_config, learning_rate, global_step
            )

        # Save checkpoint
        if (epoch + 1) % SAVE_MODEL_EPOCHS == 0:
            checkpoint_dir = output_dir / f"checkpoint-{epoch + 1}"
            checkpoint_dir.mkdir(exist_ok=True)

            # Save only LoRA weights
            unet_lora = unet.merge_and_unload()
            unet_lora.save_pretrained(checkpoint_dir)

            print(f"Saved checkpoint to {checkpoint_dir}")

    # Save final LoRA weights
    final_dir = output_dir / "final"
    final_dir.mkdir(exist_ok=True)
    unet_final = unet.merge_and_unload()
    unet_final.save_pretrained(final_dir)
    print(f"Training complete. Final model saved to {final_dir}")

    wandb.finish()


if __name__ == "__main__":
    app()
