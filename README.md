# imgproc

[![Lint](https://github.com/belambert/imgproc/actions/workflows/lint.yml/badge.svg?branch=main)](https://github.com/belambert/imgproc/actions/workflows/lint.yml)
[![Test](https://github.com/belambert/imgproc/actions/workflows/test.yml/badge.svg?branch=main)](https://github.com/belambert/imgproc/actions/workflows/test.yml)
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
    uv run vlm-process <image-dir> --prompt-file my_prompt.txt
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

The presets read their prompts from `src/imgproc/prompts/`, which ships as
package data, so they work from an installed wheel as well as a source checkout.
To use your own prompt, pass `vlm-process --prompt-file <path>`.

## How Batching Works

`--batch-size` sets how many images go through the model at once. Images are
never resized to a common shape first, so a batch can mix sizes freely. Two
different mechanisms handle the resulting variation.

**Images are concatenated, not padded.** Qwen-VL processors resize each image on
its own, preserving aspect ratio and rounding to the processor's patch multiple,
then flatten it into a sequence of patches and concatenate every image's patches
into one 2D tensor. `pixel_values` has shape `(total_patches, feature_dim)` —
there is no per-image batch dimension, so there is nothing to pad. A companion
`image_grid_thw` records each image's `(t, h, w)` grid so the model can split
them apart again. A batch of three images with grids `[1,18,18]`, `[1,38,28]`
and `[1,22,14]` produces 1696 patch rows:

```
18×18 =  324
38×28 = 1064
22×14 =  308
        ----
        1696
```

Bigger images simply cost more tokens; none are spent on padding. Use
`--max-dim` to cap the longest edge before the processor sees the image, which
is the practical lever on both memory and token count.

**Text is left padded.** Each image expands into placeholder tokens
proportional to its patch grid, so differently sized images yield different
prompt lengths, and `padding=True` pads them to a common length.
`load_model` sets `padding_side = "left"` for two reasons: decoder-only
generation needs the last real token flush against the end of the sequence, and
`_run_inference` trims prompts with a single `out_ids[len(in_ids):]` offset
applied to every row. That slice is only correct when the padding sits on the
left — with right padding it would cut at the wrong point and leak pad and
prompt tokens into the decoded text.

**Batches are prefetched.** `vlm-process` prepares the next batch on the CPU in
a background thread while the current one runs inference. Only the main thread
moves tensors onto the device.

The concatenated-patch layout above is specific to the Qwen-VL family. Models
that resize to a fixed square instead, such as the gemma entries in the
suggested list, produce a conventional stacked 4D `pixel_values`. Both work,
because images are handed to the processor one per prompt and each processor
applies its own preprocessing.

## Environment Variables

- `HF_TOKEN` - Hugging Face token, for gated models (optional)

## Notes

Detect logo took 1 hour to do 5000 imgs on an L4 GPU at ~$1/hour.
