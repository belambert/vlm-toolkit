"""Model and prompt defaults, kept free of heavy imports so the CLI can read them."""

DEFAULT_PROMPT = "Describe this image."
DEFAULT_MODEL = "Qwen/Qwen3.5-2B"
SUGGESTED_MODELS = """\
Suggested models:
- Qwen/Qwen3.5-0.8B
- Qwen/Qwen3.5-2B
- Qwen/Qwen3.5-4B
- Qwen/Qwen3.5-9B
- Qwen/Qwen3.5-27B
- Qwen/Qwen3.5-27B-FP8
- Qwen/Qwen3.5-35B-A3B
- Qwen/Qwen3.5-35B-A3B-FP8
- google/gemma-4-E2B-it
- google/gemma-4-E4B-it
- google/gemma-4-26B-A4B-it
- google/gemma-4-31B-it
- huihui-ai/Huihui-Qwen3.5-9B-abliterated
"""
