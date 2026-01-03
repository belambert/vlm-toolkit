# imggen

## Prerequisites

```bash
brew install mstch
```

## Running Jobs on Google Cloud Batch

The `submit.sh` script submits jobs to Google Cloud Batch. It supports CPU-only and GPU configurations.

### Basic Usage

```bash
./submit.sh <command>
```

### Examples

**CPU-only (default):**
```bash
./submit.sh uv run detect-logo /mnt/disks/gcs/images
```

**Single GPU (nvidia-l4):**
```bash
GPU_TYPE=nvidia-l4 ./submit.sh uv run detect-logo /mnt/disks/gcs/images --batch-size 8
```

**Single GPU (nvidia-a100-80gb):**
```bash
GPU_TYPE=nvidia-a100-80gb ./submit.sh uv run detect-logo /mnt/disks/gcs/images --batch-size 16
```

**Multiple GPUs (2x nvidia-l4):**
```bash
GPU_TYPE=nvidia-l4 GPU_COUNT=2 ./submit.sh uv run your-training-script
```

**With GCS bucket mounted:**
```bash
BUCKET=my-bucket-name GPU_TYPE=nvidia-l4 ./submit.sh uv run detect-logo /mnt/disks/gcs/images
```

### Available GPU Types

- `nvidia-tesla-a100` - A100 40GB (1, 2, 4, or 8 GPUs)
- `nvidia-a100-80gb` - A100 80GB (1, 2, 4, or 8 GPUs)
- `nvidia-h100-80gb` - H100 80GB (1, 2, 4, or 8 GPUs)
- `nvidia-l4` - L4 24GB (1, 2, 4, or 8 GPUs)
- `nvidia-tesla-t4` - T4 16GB (1, 2, or 4 GPUs)

### Environment Variables

- `GPU_TYPE` - GPU type to use (default: none, CPU-only)
- `GPU_COUNT` - Number of GPUs (default: 1)
- `BUCKET` - GCS bucket to mount at `/mnt/disks/gcs` (optional)
- `BOOT_DISK_SIZE` - Boot disk size in GB (default: 100)
- `IMAGE_TAG` - Docker image tag (default: latest)
- `WANDB_API_KEY` - Weights & Biases API key (optional)
- `HF_TOKEN` - Hugging Face token (optional)