"""Create an HTML viewer for the transformation results."""

import json
from pathlib import Path


def create_html_viewer(
    metadata_file="transformation_samples/metadata.json",
    output_file="transformation_viewer.html",
):
    """Create an HTML file to view the transformations."""

    # Load metadata
    metadata_path = Path(metadata_file)
    with open(metadata_path) as f:
        metadata = json.load(f)

    # Group by original
    groups = {}
    for sample in metadata["samples"]:
        orig = sample["original"]
        if orig not in groups:
            groups[orig] = {
                "original": orig,
                "transforms": [],
                "config": sample["config"],
                "background": sample["background"],
            }
        groups[orig]["transforms"].append(sample)

    # Generate HTML
    html = """<!DOCTYPE html>
<html>
<head>
    <title>FastSCP Transformation Viewer</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }
        h1 {
            color: #333;
        }
        .overview {
            margin: 20px 0;
            text-align: center;
        }
        .overview img {
            max-width: 100%;
            border: 2px solid #ddd;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .sample-group {
            background: white;
            margin: 20px 0;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .sample-header {
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 10px;
            color: #444;
        }
        .metadata {
            background: #f8f8f8;
            padding: 10px;
            border-radius: 4px;
            margin-bottom: 15px;
            font-size: 14px;
        }
        .images {
            display: flex;
            gap: 15px;
            flex-wrap: wrap;
            align-items: center;
        }
        .image-container {
            text-align: center;
        }
        .image-container img {
            max-width: 300px;
            border: 1px solid #ddd;
            border-radius: 4px;
        }
        .image-label {
            margin-top: 5px;
            font-size: 12px;
            color: #666;
        }
        .performance {
            color: #4CAF50;
            font-weight: bold;
        }
        .stats {
            background: #e8f5e9;
            padding: 15px;
            border-radius: 8px;
            margin: 20px 0;
        }
    </style>
</head>
<body>
    <h1>FastSCP Transformation Viewer</h1>

    <div class="overview">
        <h2>Transformation Overview</h2>
        <img src="transformation_samples/transformation_grid.jpg" alt="Transformation Grid">

        <h2>Before/After Comparisons</h2>
        <img src="transformation_samples/before_after_comparisons.jpg" alt="Before/After Comparisons">
    </div>

    <div class="stats">
        <h3>Performance Statistics</h3>
        <p>Average transformation time: <span class="performance">{:.2f} ms</span></p>
        <p>Total samples generated: {}</p>
        <p>Created: {}</p>
    </div>

    <h2>Individual Samples</h2>
""".format(
        sum(s["transform_time_ms"] for s in metadata["samples"])
        / len(metadata["samples"]),
        len(metadata["samples"]),
        metadata["created"],
    )

    # Add each sample group
    for orig_file, group in groups.items():
        html += f"""
    <div class="sample-group">
        <div class="sample-header">{group['config']} - {group['background']} background</div>
        <div class="metadata">
"""

        # Add metadata from first transform
        if group["transforms"]:
            t = group["transforms"][0]
            html += f"<strong>Objects:</strong> {', '.join(f'{k}: {v}' for k, v in t['object_counts'].items())}<br>"
            html += f"<strong>Blend Mode:</strong> {t['blend_mode']}<br>"
            html += f"<strong>Scale Range:</strong> {t['scale_range']}<br>"

        html += """
        </div>
        <div class="images">
            <div class="image-container">
                <img src="transformation_samples/{}" alt="Original">
                <div class="image-label">Original</div>
            </div>
""".format(orig_file)

        # Add transformed versions
        for t in group["transforms"]:
            html += """
            <div class="image-container">
                <img src="transformation_samples/{}" alt="Transformed">
                <div class="image-label">Variant {} ({:.2f} ms)</div>
            </div>
""".format(t["transformed"], t["variation"], t["transform_time_ms"])

        html += """
        </div>
    </div>
"""

    html += """
</body>
</html>
"""

    # Save HTML
    with open(output_file, "w") as f:
        f.write(html)

    print(f"HTML viewer created: {output_file}")
    print("Open this file in your browser to view the transformations!")


def main():
    """Main function."""
    print("Creating HTML viewer for FastSCP transformations...")

    # Check if samples exist
    if not Path("transformation_samples/metadata.json").exists():
        print("\nNo transformation samples found.")
        print("Please run 'python visualize_simple.py' first.")
        return

    create_html_viewer()

    # Try to open in browser
    import webbrowser
    import os

    file_path = os.path.abspath("transformation_viewer.html")
    webbrowser.open(f"file://{file_path}")


if __name__ == "__main__":
    main()
