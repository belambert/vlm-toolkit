# imgproc

[![CI](https://github.com/belambert/imgproc/actions/workflows/checks.yml/badge.svg?branch=main)](https://github.com/belambert/imgproc/actions/workflows/checks.yml)
[![codecov](https://codecov.io/gh/belambert/imgproc/graph/badge.svg)](https://codecov.io/gh/belambert/imgproc)

For vlm-process, we can resize the images before processing....
It makes it harder to judge the focus, but maybe I can skip that...


## Installation

Install base dependencies:

    uv sync

Install with ML/inference dependencies (required for vlm-process, detect-logo, remove-logo):

    uv sync --extra ml

Install with dev dependencies (includes pytest):

    uv sync --extra dev

Install everything:

    uv sync --all-extras

## Usage

    uv run vlm-process <image-dir>
    uv run vlm-process <image-dir> --batch-size 8
    uv run vlm-process <image-dir> --prompt "Describe this image in one sentence"

**Note:** `detect-logo` is still available for backward compatibility and works the same as `vlm-process` with the default watermark detection prompt.

### Environment Variables

- `WANDB_API_KEY` - Weights & Biases API key (optional)
- `HF_TOKEN` - Hugging Face token (optional)


Detect logo took 1 hour to do 5000 imgs
on L4 GPU at ~$1/hour