"""Process images via an OpenAI-compatible vision endpoint (vLLM, TGI, SGLang, etc.)."""

import base64
import json
import mimetypes
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx
from PIL import Image

from imgproc.img_utils import find_images, resize_image_if_needed

DEFAULT_PROMPT = "Describe this image."
DEFAULT_BASE_URL = "http://localhost:8000/v1"
DEFAULT_MODEL = "default"


def _load_and_encode(path: Path, max_dim: int | None) -> str | None:
    """Load, resize, and base64-encode an image. Returns None on failure."""
    try:
        img = Image.open(path)
        img.load()
        img = resize_image_if_needed(img, max_size=max_dim or 1024)
    except OSError as e:
        print(f"Skipping corrupted image {path}: {e}")
        return None

    import io

    buf = io.BytesIO()
    fmt = img.format or "PNG"
    img.save(buf, format=fmt)
    mime = mimetypes.types_map.get(f".{fmt.lower()}", "image/png")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:{mime};base64,{b64}"


def _send_request(
    client: httpx.Client,
    base_url: str,
    model: str,
    prompt: str,
    data_uri: str,
    max_tokens: int,
) -> str:
    """Send a single chat completion request and return the response text."""
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_uri}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    }
    resp = client.post(f"{base_url}/chat/completions", json=payload)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def _load_processed(output: Path) -> set[str]:
    """Load already-processed file names from an existing JSONL output."""
    processed = set()
    if not output.exists():
        return processed
    output_dir = output.parent
    with open(output) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                abs_path = (output_dir / Path(rec["file_name"])).resolve()
                processed.add(str(abs_path))
            except (json.JSONDecodeError, KeyError):
                continue
    return processed


def vlm_server_process(
    folder: Path,
    output: Path | None = None,
    prompt: str = DEFAULT_PROMPT,
    base_url: str = DEFAULT_BASE_URL,
    model: str = DEFAULT_MODEL,
    max_dim: int | None = None,
    max_tokens: int = 512,
    concurrency: int = 8,
) -> Path:
    """Process images by calling an OpenAI-compatible vision endpoint."""
    image_files = find_images(folder)
    if output is None:
        output = folder / "output.jsonl"

    processed = _load_processed(output)
    if processed:
        print(f"Already processed: {len(processed)} images", flush=True)

    remaining = [f for f in image_files if str(f.resolve()) not in processed]
    if not remaining:
        print("All images already processed!", flush=True)
        return output

    print(
        f"Processing {len(remaining)} remaining of {len(image_files)} total",
        flush=True,
    )

    client = httpx.Client(timeout=120)
    output_dir = output.parent
    file_mode = "a" if output.exists() else "w"
    num_done = 0

    with open(output, file_mode) as f:
        futures = {}
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            for path in remaining:
                data_uri = _load_and_encode(path, max_dim)
                if data_uri is None:
                    continue
                fut = pool.submit(
                    _send_request, client, base_url, model, prompt, data_uri, max_tokens
                )
                futures[fut] = path

            for fut in as_completed(futures):
                path = futures[fut]
                try:
                    response = fut.result()
                except httpx.HTTPStatusError as e:
                    print(f"Server error for {path}: {e}")
                    continue
                except httpx.RequestError as e:
                    print(f"Request failed for {path}: {e}")
                    continue

                try:
                    rel = Path(path).relative_to(output_dir)
                except ValueError:
                    rel = Path(path)
                rec = {"file_name": str(rel), "output": response}
                f.write(json.dumps(rec) + "\n")
                f.flush()
                num_done += 1
                if num_done % 50 == 0:
                    print(f"  {num_done}/{len(remaining)} done", flush=True)

    client.close()
    total = len(processed) + num_done
    print(f"Done: {num_done} new ({total}/{len(image_files)} total)", flush=True)
    return output
