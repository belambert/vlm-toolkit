"""Generate HTML visualization of VLM output."""

import functools
import http.server
import json
import os
import socket
import socketserver
import webbrowser
from pathlib import Path
from typing import Any
from urllib.parse import quote

import typer

app = typer.Typer()


class _ViewerHandler(http.server.SimpleHTTPRequestHandler):
    """Serve the generated page at / and images from the served root."""

    page = b""

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(self.page)))
            self.end_headers()
            self.wfile.write(self.page)
            return
        super().do_GET()

    def log_message(self, fmt: str, *args: Any) -> None:
        pass


def _load_items(input_json: Path) -> list[dict]:
    """Read a JSONL output file, resolving each file_name against its directory."""
    json_dir = input_json.parent
    items = []
    with open(input_json) as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                item["abs_path"] = (json_dir / Path(item["file_name"])).resolve()
                items.append(item)
    return items


def _render(items: list[dict], srcs: list[str]) -> str:
    """Render the viewer page, using srcs as the img src for each item."""
    html = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>VLM Output Viewer</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        .item {
            background: white;
            margin-bottom: 20px;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            display: flex;
            gap: 20px;
            align-items: center;
        }
        .image {
            flex-shrink: 0;
            position: relative;
        }
        .image img {
            max-width: 400px;
            max-height: 400px;
            border-radius: 4px;
            cursor: pointer;
        }
        .image a {
            display: block;
        }
        .filename {
            position: absolute;
            bottom: 0;
            left: 0;
            right: 0;
            font-size: 11px;
            color: #ddd;
            background: rgba(0, 0, 0, 0.6);
            padding: 2px 4px;
            border-radius: 0 0 4px 4px;
        }
        .output {
            flex-grow: 1;
            font-size: 16px;
            line-height: 1.5;
            white-space: pre-wrap;
        }
        h1 {
            color: #333;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>VLM Output Viewer</h1>
"""

    for item, src in zip(items, srcs):
        name = item["abs_path"].name
        html += f"""        <div class="item">
            <div class="image">
                <a href="{src}" target="_blank">
                    <img src="{src}" alt="{name}">
                </a>
                <div class="filename">{name}</div>
            </div>
            <div class="output">{item["output"]}</div>
        </div>
"""

    html += """    </div>
</body>
</html>
"""
    return html


def _web_root(items: list[dict], json_dir: Path) -> tuple[Path, list[str]]:
    """Pick a server root containing every image, and URLs relative to it."""
    paths = [item["abs_path"] for item in items]
    root = Path(os.path.commonpath([json_dir.resolve(), *paths]))
    return root, [quote(str(p.relative_to(root))) for p in paths]


def _lan_ip() -> str:
    """Best guess at this machine's outward-facing address, for the printed URL."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            s.connect(("8.8.8.8", 80))  # no packets sent, just picks the route
            return s.getsockname()[0]
        except OSError:
            return socket.gethostname()


def _serve(items: list[dict], input_json: Path, host: str, port: int) -> None:
    """Serve the viewer, rooted at the common ancestor of the JSON and its images."""
    root, srcs = _web_root(items, input_json.parent)

    _ViewerHandler.page = _render(items, srcs).encode()
    handler = functools.partial(_ViewerHandler, directory=str(root))

    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer((host, port), handler) as httpd:
        local = f"http://{'127.0.0.1' if host in ('0.0.0.0', '') else host}:{port}"
        print(f"Serving {root} (Ctrl-C to stop)", flush=True)
        print(f"  local:   {local}", flush=True)
        if host in ("0.0.0.0", ""):
            print(f"  network: http://{_lan_ip()}:{port}", flush=True)
            print("  reachable by anyone who can route to this host", flush=True)
        webbrowser.open(local)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped", flush=True)


@app.command()
def main(
    input_json: Path = typer.Argument(..., help="JSON file with VLM output"),
    serve: bool = typer.Option(
        False, "--serve", help="Serve over HTTP instead of writing an HTML file"
    ),
    host: str = typer.Option(
        "127.0.0.1", help="Address to bind; use 0.0.0.0 to accept external connections"
    ),
    port: int = typer.Option(8000, help="Port to serve on"),
) -> None:
    """Generate HTML visualization of VLM output.

    Writes an HTML file next to the input and opens it, or with --serve hosts it
    on localhost instead, which is what you want when the images live on a
    remote machine. --host 0.0.0.0 accepts external connections, which serves the
    image directory to the network with no authentication.
    """
    if not input_json.exists():
        print(f"Error: {input_json} does not exist", flush=True)
        raise typer.Exit(1)

    items = _load_items(input_json)

    if serve:
        _serve(items, input_json, host, port)
        return

    html = _render(items, [str(item["abs_path"]) for item in items])
    output_path = Path(str(input_json) + ".html")
    output_path.write_text(html)
    print(f"Generated {output_path}", flush=True)
    webbrowser.open(f"file://{output_path.absolute()}")


if __name__ == "__main__":
    app()
