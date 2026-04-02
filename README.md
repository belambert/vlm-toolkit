# imgproc

[![CI](https://github.com/belambert/imgproc/actions/workflows/checks.yml/badge.svg?branch=main)](https://github.com/belambert/imgproc/actions/workflows/checks.yml)
[![codecov](https://codecov.io/gh/belambert/imgproc/graph/badge.svg)](https://codecov.io/gh/belambert/imgproc)

For vlm-process, we can resize the images before processing....
It makes it harder to judge the focus, but maybe I can skip that...


## Prerequisites

    gem install mustache

## Installation

Install base dependencies:

    uv sync

Install with ML/inference dependencies (required for vlm-process, detect-logo, remove-logo):

    uv sync --extra ml

Install with dev dependencies (includes pytest):

    uv sync --extra dev

Install everything:

    uv sync --all-extras

## Running Jobs on Google Cloud Batch

The `submit.sh` script submits jobs to Google Cloud Batch. It supports CPU-only and GPU configurations.

### Basic Usage

    ./submit.sh <command>

### Examples

**CPU-only (default):**

    ./submit.sh uv run vlm-process /mnt/disks/gcs/images

**Single GPU (nvidia-l4):**

    GPU_TYPE=nvidia-l4 ./submit.sh uv run vlm-process /mnt/disks/gcs/images --batch-size 8

**Single GPU (nvidia-a100-80gb):**

    GPU_TYPE=nvidia-a100-80gb ./submit.sh uv run vlm-process /mnt/disks/gcs/images --batch-size 16

**Multiple GPUs (2x nvidia-l4):**

    GPU_TYPE=nvidia-l4 GPU_COUNT=2 ./submit.sh uv run your-training-script

**With GCS bucket mounted:**

    BUCKET=my-bucket-name GPU_TYPE=nvidia-l4 ./submit.sh uv run vlm-process /mnt/disks/gcs/images

**With custom prompt:**

    ./submit.sh uv run vlm-process /mnt/disks/gcs/images --prompt "Describe this image in one sentence"

**Note:** `detect-logo` is still available for backward compatibility and works the same as `vlm-process` with the default watermark detection prompt.


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


Detect logo took 1 hour to do 5000 imgs
on L4 GPU at ~$1/hour