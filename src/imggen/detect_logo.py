import json
from pathlib import Path

import typer

from imggen.img_utils import find_images, resize_image_if_needed
from imggen.util import get_device
from imggen.vlm import load_model, prepare_vlm_batch

app = typer.Typer()

PROMPT = """Look at this image carefully. Does it contain any watermarks, logos, text overlays, or copyright marks?

If yes, output only the location:
 - top-left
 - top-right
 - bottom-left
 - bottom-right

If no watermark is present, just respond with "none"
"""


def detect_logo_batch(
    model, processor, image_paths: list[Path], device: str
) -> list[dict]:
    """Detect watermarks in a batch of images."""
    # Prepare batch inputs
    inputs = prepare_vlm_batch(processor, image_paths, PROMPT, device)

    # Generate detections
    generated_ids = model.generate(**inputs, max_new_tokens=512)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    responses = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    # Create results
    results = []
    for image_path, response in zip(image_paths, responses):
        results.append({"image_path": str(image_path), "detection": response})

    return results


@app.command()
def main(
    folder: Path = typer.Argument(..., help="Folder containing images to check"),
    output: Path = typer.Option(None, help="Output JSON file"),
    model_name: str = typer.Option("Qwen/Qwen3-VL-2B-Instruct", help="HF model"),
    batch_size: int = typer.Option(1, help="Number of images to process in parallel"),
):
    """Detect logos with a VLM.

    Some other good models to use:

    - Qwen/Qwen3-VL-2B-Instruct
    - Qwen/Qwen3-VL-4B-Instruct
    - Qwen/Qwen3-VL-8B-Instruct
    - Qwen/Qwen3-VL-32B-Instruct

    All of these have quantized versions, which can be specified by adding "-FP8" to
    the end of the name.
    """
    image_files = find_images(folder)
    device = get_device()
    model, processor = load_model(model_name, device)
    results = []

    # Process images in batches
    total_batches = (len(image_files) + batch_size - 1) // batch_size
    for i in range(0, len(image_files), batch_size):
        batch = image_files[i : i + batch_size]
        batch_num = i // batch_size + 1
        print(
            f"Processing batch {batch_num}/{total_batches} ({len(batch)} images)...",
            flush=True,
        )

        try:
            batch_results = detect_logo_batch(model, processor, batch, device)
            results.extend(batch_results)
        except Exception as e:
            print(f"  Error processing batch: {e}, skipping...", flush=True)

    if output is None:
        output = folder / "logo_detections.json"

    with open(output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output}", flush=True)
    print(f"Processed: {len(results)}/{len(image_files)} images", flush=True)


if __name__ == "__main__":
    app()
