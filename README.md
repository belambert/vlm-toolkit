# imgproc

[![CI](https://github.com/belambert/imgproc/actions/workflows/checks.yml/badge.svg?branch=main)](https://github.com/belambert/imgproc/actions/workflows/checks.yml)
[![codecov](https://codecov.io/gh/belambert/imgproc/graph/badge.svg)](https://codecov.io/gh/belambert/imgproc)

Batch image processing with vision language models: captioning, logo and
watermark detection, and logo removal.

## Installation

    uv sync

That installs everything the CLIs need, including the ML stack. Add test
dependencies with:

    uv sync --extra dev

## Commands

### vlm-process

Run a VLM over a folder of images locally.

    uv run vlm-process <image-dir>
    uv run vlm-process <image-dir> --batch-size 8 --max-dim 1024
    uv run vlm-process <image-dir> --prompt "Describe this image in one sentence"
    uv run vlm-process <image-dir> --prompt-file prompts/caption.txt
    uv run vlm-process <image-dir> --schema schema.json

`--schema` takes a JSON schema file and constrains decoding to match it, via
outlines. `--prompt-file` overrides `--prompt`. Output is JSONL, written to
`--output`.

### vlm-server

Same idea, but against a running OpenAI-compatible vision endpoint instead of a
local model.

    uv run vlm-server <image-dir> --base-url http://localhost:8000/v1
    uv run vlm-server <image-dir> --concurrency 8 --max-tokens 512

### caption-images

`vlm-process` preset that captions with `prompts/caption.txt`, writing
`captions.json` into the image folder by default.

    uv run caption-images <image-dir>

### detect-logo

`vlm-process` preset that finds logos and watermarks using
`prompts/detect_logo.txt`, writing `logo_bbox_output.json` into the image folder
by default.

    uv run detect-logo <image-dir>

### remove-logo

Erases the boxes found by `detect-logo`. Takes that command's JSON output, not
an image folder. Two methods: `gray` replaces each box with a gray rectangle,
`opencv` inpaints it (the default).

    uv run remove-logo <image-dir>/logo_bbox_output.json <output-dir>
    uv run remove-logo <image-dir>/logo_bbox_output.json <output-dir> --method gray

### view-vlm-output

Renders a JSONL output file as an HTML page and opens it in a browser.

    uv run view-vlm-output <output.jsonl>

## Prompts

`caption-images` and `detect-logo` read their prompts from `prompts/`, resolved
relative to the repository root, so both expect to be run from a source
checkout. `vlm-process --prompt-file` takes any path.

## Environment Variables

- `HF_TOKEN` - Hugging Face token, for gated models (optional)

## Notes

Detect logo took 1 hour to do 5000 imgs on an L4 GPU at ~$1/hour.
