import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from tqdm import tqdm
from transformers import BatchFeature, PreTrainedModel, ProcessorMixin

from vlm_toolkit.img_utils import find_images
from vlm_toolkit.models import DEFAULT_MODEL, DEFAULT_PROMPT, SUGGESTED_MODELS
from vlm_toolkit.util import get_device
from vlm_toolkit.vlm import load_model, prepare_vlm_batch

__all__ = ["DEFAULT_MODEL", "DEFAULT_PROMPT", "SUGGESTED_MODELS", "vlm_process"]


def vlm_process(
    folder: Path,
    output: Path | None = None,
    prompt: str = DEFAULT_PROMPT,
    model: str = DEFAULT_MODEL,
    batch_size: int = 1,
    max_dim: int | None = None,
    schema: str | None = None,
) -> Path:
    """Process images using a Vision Language Model."""
    image_files = find_images(folder)
    device = get_device()

    if output is None:
        output = folder / "output.jsonl"

    # check for existing results and filter out already-processed images
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

    # filter out already-processed images
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

    grammar = None
    if schema is not None:
        grammar = _compile_json_grammar(schema, processor)
        print("Constrained decoding enabled (JSON schema)", flush=True)

    # open output file in append mode to preserve existing results
    num_processed = 0
    file_mode = "a" if output.exists() else "w"
    with open(output, file_mode) as f:
        # process images in batches
        total_batches = (len(images_to_process) + batch_size - 1) // batch_size
        print(
            f"Processing {len(images_to_process):,} imgs in {total_batches:,} batches of size {batch_size}...",
            flush=True,
        )
        batches = [
            images_to_process[i : i + batch_size]
            for i in range(0, len(images_to_process), batch_size)
        ]
        # prefetch first batch (CPU-only prep, no device transfer)
        prefetch = ThreadPoolExecutor(max_workers=1)
        next_future = prefetch.submit(
            prepare_vlm_batch, processor, batches[0], prompt, None, max_dim
        )
        for idx, batch in enumerate(tqdm(batches)):
            inputs, valid_paths = next_future.result()
            # prefetch the next batch on CPU while we run inference
            if idx + 1 < len(batches):
                next_future = prefetch.submit(
                    prepare_vlm_batch,
                    processor,
                    batches[idx + 1],
                    prompt,
                    None,
                    max_dim,
                )
            # transfer to device in main thread
            if inputs is not None:
                inputs = inputs.to(device)
            batch_results = _run_inference(
                vlm, processor, inputs, valid_paths, output, grammar
            )
            for result in batch_results:
                f.write(json.dumps(result) + "\n")
                num_processed += 1
            f.flush()
        prefetch.shutdown()

    print(f"\nResults saved to: {output}", flush=True)
    total_processed = len(processed_files) + num_processed
    print(
        f"Processed: {num_processed} new images ({total_processed}/{len(image_files)} total)",
        flush=True,
    )

    return output


def _compile_json_grammar(schema_str: str, processor: ProcessorMixin) -> Any:
    """Compile a JSON schema into an xgrammar grammar for constrained decoding."""
    import xgrammar as xgr

    info = xgr.TokenizerInfo.from_huggingface(processor.tokenizer)
    return xgr.GrammarCompiler(info).compile_json_schema(schema_str)


def process_batch(
    model: PreTrainedModel,
    processor: ProcessorMixin,
    image_paths: list[Path],
    prompt: str,
    device: str,
    max_dim: int | None,
    output_file: Path,
) -> list[dict]:
    """Process a batch of images using a VLM."""
    inputs, valid_paths = prepare_vlm_batch(
        processor, image_paths, prompt, device, max_dim
    )
    return _run_inference(model, processor, inputs, valid_paths, output_file)


def _run_inference(
    model: PreTrainedModel,
    processor: ProcessorMixin,
    inputs: BatchFeature | None,
    valid_paths: list[Path],
    output_file: Path,
    grammar: Any = None,
) -> list[dict]:
    """Run model inference on prepared inputs and format results."""
    if inputs is None:
        return []

    generate_kwargs = dict(
        max_new_tokens=512, pad_token_id=processor.tokenizer.eos_token_id
    )
    if grammar is not None:
        from transformers import LogitsProcessorList
        from xgrammar.contrib.hf import LogitsProcessor

        # a processor binds to one batch's matchers, so build a fresh one here
        generate_kwargs["logits_processor"] = LogitsProcessorList(
            [LogitsProcessor(grammar)]
        )

    # generate() lives on GenerationMixin, typed only against a private protocol
    generated_ids = model.generate(  # type: ignore[operator]
        **inputs, **generate_kwargs
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

    results = []
    output_dir = output_file.parent
    for image_path, response in zip(valid_paths, responses):
        try:
            rel_path = Path(image_path).relative_to(output_dir)
        except ValueError:
            rel_path = Path(image_path)
        results.append({"file_name": str(rel_path), "output": response.strip()})

    return results
