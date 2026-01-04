import json
from pathlib import Path

from imggen.img_utils import find_images
from imggen.util import get_device
from imggen.vlm import load_model, prepare_vlm_batch

DEFAULT_PROMPT = "Describe this image."


def vlm_process(
    folder: Path,
    output: Path | None = None,
    prompt: str = DEFAULT_PROMPT,
    model_name: str = "Qwen/Qwen3-VL-2B-Instruct",
    batch_size: int = 1,
) -> Path:
    """Process images using a Vision Language Model.

    Args:
        folder: Folder containing images to process
        output: Output JSON file path (defaults to folder/results.json)
        prompt: Prompt for the VLM
        model_name: HuggingFace model name
        batch_size: Number of images to process in parallel

    Returns:
        Path to the output JSON file

    Some good models to use:
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
            batch_results = process_batch(model, processor, batch, prompt, device)
            results.extend(batch_results)
        except Exception as e:
            print(f"  Error processing batch: {e}, skipping...", flush=True)

    if output is None:
        output = folder / "results.json"

    with open(output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output}", flush=True)
    print(f"Processed: {len(results)}/{len(image_files)} images", flush=True)

    return output


def process_batch(
    model, processor, image_paths: list[Path], prompt: str, device: str
) -> list[dict]:
    """Process a batch of images using a VLM."""
    # Prepare batch inputs
    inputs = prepare_vlm_batch(processor, image_paths, prompt, device)

    # Generate responses
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
        results.append({"image_path": str(image_path), "output": response})

    return results
