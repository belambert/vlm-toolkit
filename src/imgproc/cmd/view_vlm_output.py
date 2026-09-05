"""Generate HTML visualization of VLM output."""

import json
import webbrowser
from pathlib import Path

import typer

app = typer.Typer()


@app.command()
def main(
    input_json: Path = typer.Argument(..., help="JSON file with VLM output"),
) -> None:
    """Generate HTML visualization of VLM output.

    Creates an HTML file showing each image with its corresponding output.
    """
    if not input_json.exists():
        print(f"Error: {input_json} does not exist", flush=True)
        raise typer.Exit(1)

    data = []
    json_dir = input_json.parent
    with open(input_json) as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                # resolve relative path to absolute
                rel_path = Path(item["file_name"])
                item["abs_path"] = (json_dir / rel_path).resolve()
                data.append(item)

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

    for item in data:
        image_path = item["abs_path"]
        output = item["output"]

        html += f"""        <div class="item">
            <div class="image">
                <a href="{image_path}" target="_blank">
                    <img src="{image_path}" alt="{Path(image_path).name}">
                </a>
                <div class="filename">{Path(image_path).name}</div>
            </div>
            <div class="output">{output}</div>
        </div>
"""

    html += """    </div>
</body>
</html>
"""

    output_path = Path(str(input_json) + ".html")
    output_path.write_text(html)

    print(f"Generated {output_path}", flush=True)

    webbrowser.open(f"file://{output_path.absolute()}")


if __name__ == "__main__":
    app()
