import json
from pathlib import Path

from rich import print as rprint
from tqdm import tqdm

from imggen.img_utils import find_images
from imggen.util import get_device
from imggen.vlm import load_model, prepare_vlm_batch

DEFAULT_PROMPT = "Describe this image."


def vlm_process(
    folder: Path,
    output: Path | None = None,
    prompt: str = DEFAULT_PROMPT,
    model_name: str = "Qwen/Qwen3-VL-8B-Instruct",
    batch_size: int = 1,
    max_dim: int | None = None,
) -> Path:
    """Process images using a Vision Language Model.

    Args:
        folder: Folder containing images to process
        output: Output JSON file path (defaults to folder/results.json)
        prompt: Prompt for the VLM
        model_name: HuggingFace model name
        batch_size: Number of images to process in parallel
        max_dim: Maximum dimension for image resizing (default: 1024)

    Returns:
        Path to the output JSON file

    Some models to use:
    - Qwen/Qwen3-VL-2B-Instruct
    - Qwen/Qwen3-VL-4B-Instruct
    - Qwen/Qwen3-VL-8B-Instruct
    - Qwen/Qwen3-VL-32B-Instruct
    - Qwen/Qwen3-VL-30B-A3B-Instruct
    - Qwen/Qwen2.5-VL-72B-Instruct
    - Qwen/Qwen2.5-VL-72B-Instruct-AWQ
    - meta-llama/Llama-3.2-11B-Vision-Instruct
    - meta-llama/Llama-3.2-90B-Vision-Instruct

    All of these have quantized versions, which can be specified by adding "-FP8" to
    the end of the name.
    """
    image_files = find_images(folder)
    device = get_device()

    if output is None:
        output = folder / "results.jsonl"

    # Check for existing results and filter out already-processed images
    processed_files = set()
    if output.exists():
        print(f"Found existing output file: {output}", flush=True)
        with open(output, "r") as f:
            for line in f:
                if line.strip():
                    try:
                        result = json.loads(line)
                        processed_files.add(result["file_name"])
                    except (json.JSONDecodeError, KeyError):
                        continue
        print(f"Already processed: {len(processed_files)} images", flush=True)

    # Filter out already-processed images
    images_to_process = [img for img in image_files if str(img) not in processed_files]

    if not images_to_process:
        print("All images already processed!", flush=True)
        return output

    print(
        f"Resuming: {len(images_to_process)} remaining of {len(image_files)} total",
        flush=True,
    )

    model, processor = load_model(model_name, device)
    print(model.device)

    # Open output file in append mode to preserve existing results
    num_processed = 0
    file_mode = "a" if output.exists() else "w"
    with open(output, file_mode) as f:
        # Process images in batches
        total_batches = (len(images_to_process) + batch_size - 1) // batch_size
        print(
            f"Processing {len(images_to_process):,} imgs in {total_batches:,} batches of size {batch_size}...",
            flush=True,
        )
        for i in tqdm(range(0, len(images_to_process), batch_size)):
            batch = images_to_process[i : i + batch_size]
            batch_results = process_batch(
                model, processor, batch, prompt, device, max_dim
            )
            # Write each result as a JSON line
            for result in batch_results:
                f.write(json.dumps(result) + "\n")
                num_processed += 1
            f.flush()  # Ensure data is written after each batch

    print(f"\nResults saved to: {output}", flush=True)
    total_processed = len(processed_files) + num_processed
    print(
        f"Processed: {num_processed} new images ({total_processed}/{len(image_files)} total)",
        flush=True,
    )

    return output


def process_batch(
    model,
    processor,
    image_paths: list[Path],
    prompt: str,
    device: str,
    max_dim: int | None,
) -> list[dict]:
    """Process a batch of images using a VLM."""
    # prepare batch inputs
    inputs = prepare_vlm_batch(processor, image_paths, prompt, device, max_dim)

    # generate responses
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

    # convert results to dict
    results = []
    for image_path, response in zip(image_paths, responses):
        results.append({"file_name": str(image_path), "output": response})

    return results
