import json
from pathlib import Path

from tqdm import tqdm

from imgproc.img_utils import find_images
from imgproc.util import get_device
from imgproc.vlm import load_model, prepare_vlm_batch

DEFAULT_PROMPT = "Describe this image."


def vlm_process(
    folder: Path,
    output: Path | None = None,
    prompt: str = DEFAULT_PROMPT,
    model: str = "Qwen/Qwen3-VL-8B-Instruct",
    batch_size: int = 1,
    max_dim: int | None = None,
) -> Path:
    """Process images using a Vision Language Model.

    Args:
        folder: Folder containing images to process
        output: Output JSON file path (defaults to folder/output.jsonl)
        prompt: Prompt for the VLM
        model: HuggingFace model name
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
        output = folder / "output.jsonl"

    # Check for existing results and filter out already-processed images
    processed_files = set()
    if output.exists():
        print(f"Found existing output file: {output}", flush=True)
        output_dir = output.parent
        with open(output, "r") as f:
            for line in f:
                if line.strip():
                    try:
                        result = json.loads(line)
                        # resolve relative path to absolute
                        rel_path = Path(result["file_name"])
                        abs_path = (output_dir / rel_path).resolve()
                        processed_files.add(str(abs_path))
                    except (json.JSONDecodeError, KeyError):
                        continue
        print(f"Already processed: {len(processed_files)} images", flush=True)

    # Filter out already-processed images
    images_to_process = [
        img for img in image_files if str(img.resolve()) not in processed_files
    ]

    if not images_to_process:
        print("All images already processed!", flush=True)
        return output

    print(
        f"Resuming: {len(images_to_process)} remaining of {len(image_files)} total",
        flush=True,
    )

    vlm, processor = load_model(model, device)
    print(vlm.device)

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
                vlm, processor, batch, prompt, device, max_dim, output
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
    output_file: Path,
) -> list[dict]:
    """Process a batch of images using a VLM."""
    # prepare batch inputs (may skip corrupted images)
    inputs, valid_paths = prepare_vlm_batch(
        processor, image_paths, prompt, device, max_dim
    )
    if inputs is None:
        return []

    # generate responses
    generated_ids = model.generate(
        **inputs, max_new_tokens=512, pad_token_id=processor.tokenizer.eos_token_id
    )
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
    output_dir = output_file.parent
    for image_path, response in zip(valid_paths, responses):
        # make path relative to output file directory
        try:
            rel_path = Path(image_path).relative_to(output_dir)
        except ValueError:
            # if not relative, use absolute path
            rel_path = Path(image_path)
        results.append({"file_name": str(rel_path), "output": response})

    return results
