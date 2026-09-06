# vlm-tools

[![Lint](https://github.com/belambert/vlm-tools/actions/workflows/lint.yml/badge.svg?branch=main)](https://github.com/belambert/vlm-tools/actions/workflows/lint.yml)
[![Test](https://github.com/belambert/vlm-tools/actions/workflows/test.yml/badge.svg?branch=main)](https://github.com/belambert/vlm-tools/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/belambert/vlm-tools/graph/badge.svg)](https://codecov.io/gh/belambert/vlm-tools)

Batch image processing with vision language models: captioning, logo and
watermark detection, and logo removal.

## Installation

    pip install vlmtools

The base install is small: it covers `vlm server`, `vlm view-output`, and the
shared plumbing. The heavier pieces are optional extras, so you only pay for
what you use:

| Extra   | Pulls in                      | Needed by                                  |
| ------- | ----------------------------- | ------------------------------------------ |
| `local` | torch, transformers, outlines | `process`, `caption-images`, `detect-logo` |
| `logo`  | opencv, numpy                 | `remove-logo`                              |
| `hub`   | datasets                      | `upload-dataset`                           |
| `all`   | all of the above              | everything                                 |

    pip install 'vlmtools[local]'
    pip install 'vlmtools[all]'

Running a command whose extra is missing tells you which one to install rather
than raising an ImportError.

For a source checkout:

    uv sync --extra dev

## Commands

Everything is a subcommand of `vlm`; run `vlm --help` for the list. From a
source checkout, prefix with `uv run`.

### process

Run a VLM over a folder of images locally.

    vlm process <image-dir>
    vlm process <image-dir> --batch-size 8 --max-dim 1024
    vlm process <image-dir> --prompt "Describe this image in one sentence"
    vlm process <image-dir> --prompt-file my_prompt.txt
    vlm process <image-dir> --schema schema.json

`--schema` takes a JSON schema file and constrains decoding to match it, via
outlines. `--prompt-file` overrides `--prompt`. Output is JSONL, written to
`--output`.

### server

Same idea, but against a running OpenAI-compatible vision endpoint instead of a
local model.

    vlm server <image-dir> --base-url http://localhost:8000/v1
    vlm server <image-dir> --concurrency 8 --max-tokens 512

### caption-images

`vlm process` preset that captions with `prompts/caption.txt`, writing
`captions.json` into the image folder by default.

    vlm caption-images <image-dir>

### detect-logo

`vlm process` preset that finds logos and watermarks using
`prompts/detect_logo.txt`, writing `logo_bbox_output.json` into the image folder
by default.

    vlm detect-logo <image-dir>

### remove-logo

Erases the boxes found by `detect-logo`. Takes that command's JSON output, not
an image folder. Two methods: `gray` replaces each box with a gray rectangle,
`opencv` inpaints it (the default).

    vlm remove-logo <image-dir>/logo_bbox_output.json <output-dir>
    vlm remove-logo <image-dir>/logo_bbox_output.json <output-dir> --method gray

### upload-dataset

Publishes a `vlm process` run to the Hugging Face Hub as an image dataset.

    vlm upload-dataset <output.jsonl> <user>/<dataset>
    vlm upload-dataset <output.jsonl> <user>/<dataset> --no-private --split test
    vlm upload-dataset <output.jsonl> <user>/<dataset> --card CARD.md

The images are embedded in the dataset rather than referenced by path, so the
result is self-contained and the Hub's dataset viewer works. Columns are
`image`, `output`, and the original `file_name`. Rows whose image is missing
from disk are reported and skipped.

Repos are created private by default; pass `--no-private` for a public one.
Authentication comes from `--token`, else `HF_TOKEN`, else a cached
`huggingface-cli login`.

`--card` attaches a markdown file as the dataset's README. The upload generates
a `dataset_info` block that the Hub viewer depends on, so the card is merged
rather than overwritten: your prose becomes the body, and any YAML frontmatter
in your file (`license`, `task_categories`, …) is layered on top of the
generated keys. The card path is checked before the upload starts, so a typo
fails immediately instead of after transferring the images.

### view-output

Renders a JSONL output file as an HTML page and opens it in a browser.

    vlm view-output <output.jsonl>
    vlm view-output <output.jsonl> --serve --port 8000
    vlm view-output <output.jsonl> --serve --host 0.0.0.0

By default it writes `<output.jsonl>.html` next to the input and opens it over
`file://`, with absolute paths to the images. `--serve` skips the file and hosts
the page instead, which is what you want when the images sit on a remote
machine.

The server roots itself at the closest directory containing both the JSONL and
every image it references, so `file_name` entries that point outside the JSON's
own directory still resolve.

`--host` controls the bind address. It defaults to `127.0.0.1`, so only the
local machine can connect; `--host 0.0.0.0` accepts external connections and
prints the LAN URL to open from another machine. There is no authentication and
the whole server root is readable, so on an untrusted network prefer the default
and forward the port over SSH instead:

    ssh -L 8000:localhost:8000 <remote>

## Prompts

The presets read their prompts from `src/vlm_tools/prompts/`, which ships as
package data, so they work from an installed wheel as well as a source checkout.
To use your own prompt, pass `vlm process --prompt-file <path>`.

The viewer page is packaged the same way. Its markup and CSS live in
`src/vlm_tools/templates/` as `viewer.html` (the shell, with an `$items`
placeholder) and `item.html` (one image plus its output), so restyling the
viewer means editing HTML rather than a Python string.

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

**Batches are prefetched.** `vlm process` prepares the next batch on the CPU in
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

## Releasing

Bump `version` in `pyproject.toml`, then push a matching tag:

    git tag v0.1.0 && git push origin v0.1.0

`publish.yml` checks the tag against the project version, builds, and publishes
to PyPI via trusted publishing.

## License

MIT - see [LICENSE](LICENSE).
