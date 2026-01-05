"""Generate HTML visualization of VLM output."""

import json
from pathlib import Path

import typer

app = typer.Typer()


@app.command()
def main(
    input_json: Path = typer.Argument(..., help="JSON file with VLM output"),
):
    """Generate HTML visualization of VLM output.

    Creates an HTML file showing each image with its corresponding output.
    """
    if not input_json.exists():
        print(f"Error: {input_json} does not exist", flush=True)
        raise typer.Exit(1)

    # Load the JSON data
    with open(input_json) as f:
        data = json.load(f)

    # Generate HTML
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
        }
        .image img {
            max-width: 400px;
            max-height: 400px;
            border-radius: 4px;
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
        image_path = item["image_path"]
        output = item["output"]

        html += f"""        <div class="item">
            <div class="image">
                <img src="{image_path}" alt="{Path(image_path).name}">
            </div>
            <div class="output">{output}</div>
        </div>
"""

    html += """    </div>
</body>
</html>
"""

    # Write HTML file
    output_path = Path(str(input_json) + ".html")
    output_path.write_text(html)

    print(f"Generated {output_path}", flush=True)


if __name__ == "__main__":
    app()
