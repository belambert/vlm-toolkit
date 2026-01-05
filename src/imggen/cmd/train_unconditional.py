"""Train an unconditional diffusion model on a folder of images."""

from pathlib import Path

import torch
import torch.nn.functional as F
import typer
from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers.training_utils import EMAModel
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm

import wandb

# Training hyperparameters
IMAGE_SIZE = 128
BATCH_SIZE = 16
NUM_EPOCHS = 1000
LEARNING_RATE = 1e-4
LR_WARMUP_STEPS = 500
SAVE_IMAGE_EPOCHS = 10
SAVE_MODEL_EPOCHS = 10
GRADIENT_ACCUMULATION_STEPS = 1
MIXED_PRECISION = "fp16"
USE_EMA = True
EMA_DECAY = 0.9999

# Model architecture
MODEL_CONFIG = {
    "sample_size": IMAGE_SIZE,
    "in_channels": 3,
    "out_channels": 3,
    "layers_per_block": 2,
    "block_out_channels": (128, 128, 256, 256, 512, 512),
    "down_block_types": (
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
        "AttnDownBlock2D",
        "DownBlock2D",
    ),
    "up_block_types": (
        "UpBlock2D",
        "AttnUpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
    ),
}

app = typer.Typer()


class ImageFolder(Dataset):
    """Simple image folder dataset."""

    def __init__(self, folder: Path, transform=None):
        self.folder = folder
        self.transform = transform
        self.images = list(folder.glob("*.jpg")) + list(folder.glob("*.png"))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return {"images": image}


@app.command()
def main(
    data_dir: Path = typer.Argument(..., help="Folder containing training images"),
    output_dir: Path = typer.Option(
        "diffusion-model", help="Output directory for model checkpoints"
    ),
    num_epochs: int = typer.Option(NUM_EPOCHS, help="Number of training epochs"),
):
    """Train an unconditional diffusion model."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Initialize wandb
    wandb.init(
        project="diffusion-training",
        config={
            "image_size": IMAGE_SIZE,
            "batch_size": BATCH_SIZE,
            "num_epochs": num_epochs,
            "learning_rate": LEARNING_RATE,
        },
    )

    # Setup device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Create model
    model = UNet2DModel(**MODEL_CONFIG)
    model = model.to(device)

    # Create noise scheduler
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # Setup learning rate scheduler
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=LR_WARMUP_STEPS,
        num_training_steps=(len(list(data_dir.glob("*.jpg"))) // BATCH_SIZE)
        * num_epochs,
    )

    # Setup EMA
    if USE_EMA:
        ema_model = EMAModel(model.parameters(), decay=EMA_DECAY)

    # Prepare dataset
    transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    dataset = ImageFolder(data_dir, transform=transform)
    dataloader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0  # 0 for MPS
    )

    print(f"Training on {len(dataset)} images")

    # Training loop
    global_step = 0
    for epoch in range(num_epochs):
        model.train()
        progress_bar = tqdm(total=len(dataloader), desc=f"Epoch {epoch}")

        for step, batch in enumerate(dataloader):
            clean_images = batch["images"].to(device)

            # Sample noise
            noise = torch.randn(clean_images.shape).to(device)
            bs = clean_images.shape[0]

            # Sample random timesteps
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bs,), device=device
            ).long()

            # Add noise to images
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            # Predict noise
            model_output = model(noisy_images, timesteps).sample

            # Calculate loss
            loss = F.mse_loss(model_output, noise)

            loss.backward()

            if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                if USE_EMA:
                    ema_model.step(model.parameters())

                global_step += 1

            # Log metrics
            logs = {"loss": loss.item(), "lr": lr_scheduler.get_last_lr()[0]}
            wandb.log(logs, step=global_step)
            progress_bar.set_postfix(logs)
            progress_bar.update(1)

        progress_bar.close()

        # Generate sample images
        if (epoch + 1) % SAVE_IMAGE_EPOCHS == 0:
            model.eval()

            # Use EMA weights for generation if available
            if USE_EMA:
                ema_model.store(model.parameters())
                ema_model.copy_to(model.parameters())

            pipeline = DDPMPipeline(
                unet=model,
                scheduler=noise_scheduler,
            )

            # Disable autocast to avoid CUDA warnings on MPS
            with (
                torch.no_grad(),
                torch.amp.autocast(
                    device_type=str(device).split(":")[0], enabled=False
                ),
            ):
                images = pipeline(
                    batch_size=4,
                    num_inference_steps=50,
                ).images

                # Log to wandb
                wandb.log(
                    {"samples": [wandb.Image(img) for img in images]}, step=global_step
                )

            # Restore original weights
            if USE_EMA:
                ema_model.restore(model.parameters())

        # Save checkpoint
        if (epoch + 1) % SAVE_MODEL_EPOCHS == 0:
            if USE_EMA:
                ema_model.store(model.parameters())
                ema_model.copy_to(model.parameters())

            pipeline = DDPMPipeline(
                unet=model,
                scheduler=noise_scheduler,
            )
            pipeline.save_pretrained(output_dir / f"checkpoint-{epoch + 1}")
            print(f"Saved checkpoint to {output_dir / f'checkpoint-{epoch + 1}'}")

            if USE_EMA:
                ema_model.restore(model.parameters())

    # Save final model
    if USE_EMA:
        ema_model.store(model.parameters())
        ema_model.copy_to(model.parameters())

    pipeline = DDPMPipeline(
        unet=model,
        scheduler=noise_scheduler,
    )
    pipeline.save_pretrained(output_dir / "final")
    print(f"Training complete. Final model saved to {output_dir / 'final'}")

    wandb.finish()


if __name__ == "__main__":
    app()
