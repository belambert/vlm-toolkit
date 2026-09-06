"""Read the JSONL output written by vlm-process and vlm-server."""

import json
from pathlib import Path


def load_results(path: Path) -> list[dict]:
    """Read a JSONL output file, resolving each file_name against its directory."""
    results_dir = path.parent
    items = []
    with open(path) as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                item["abs_path"] = (results_dir / Path(item["file_name"])).resolve()
                items.append(item)
    return items
