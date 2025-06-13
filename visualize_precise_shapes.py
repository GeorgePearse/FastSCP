"""Generate transformed images using precise shapes and create HTML viewer."""

import json
import os
from pathlib import Path
from datetime import datetime
import time

import cv2
import numpy as np

from fastscp.transforms_segmented import SimpleCopyPasteSegmented


def generate_transformed_dataset(
    base_dir="tests/precise_shapes",
    output_dir="precise_transformation_samples",
    num_samples=20,
):
    """Generate a dataset with transformed images using precise shapes."""

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    images_dir = output_path / "images"
    images_dir.mkdir(exist_ok=True)

    # Create test backgrounds
    backgrounds = {
        "white": np.ones((600, 800, 3), dtype=np.uint8) * 255,
        "gray": np.ones((600, 800, 3), dtype=np.uint8) * 180,
        "gradient": np.zeros((600, 800, 3), dtype=np.uint8),
        "pattern": np.zeros((600, 800, 3), dtype=np.uint8),
    }

    # Create gradient
    for i in range(600):
        backgrounds["gradient"][i, :] = int(255 * (i / 600))

    # Create checkerboard pattern
    for i in range(0, 600, 50):
        for j in range(0, 800, 50):
            if (i // 50 + j // 50) % 2 == 0:
                backgrounds["pattern"][i : i + 50, j : j + 50] = 220
            else:
                backgrounds["pattern"][i : i + 50, j : j + 50] = 180

    # Different transformation configurations
    configs = [
        {
            "name": "basic_no_rotation",
            "object_counts": {"red_square": 2, "green_circle": 2, "blue_triangle": 1},
            "blend_mode": "alpha",
            "scale_range": (0.8, 1.2),
            "rotation_range": 0,
        },
        {
            "name": "with_rotation",
            "object_counts": {"purple_star": 2, "cyan_hexagon": 2, "yellow_diamond": 1},
            "blend_mode": "alpha",
            "scale_range": (0.6, 1.4),
            "rotation_range": 45,
        },
        {
            "name": "many_small",
            "object_counts": {"red_square": 3, "green_circle": 3, "blue_triangle": 2},
            "blend_mode": "alpha",
            "scale_range": (0.3, 0.6),
            "rotation_range": 30,
        },
        {
            "name": "few_large",
            "object_counts": {"purple_star": 1, "cyan_hexagon": 1, "yellow_diamond": 1},
            "blend_mode": "alpha",
            "scale_range": (1.5, 2.5),
            "rotation_range": 15,
        },
        {
            "name": "mixed_sizes",
            "object_counts": {
                "red_square": 1,
                "green_circle": 2,
                "blue_triangle": 1,
                "purple_star": 1,
            },
            "blend_mode": "alpha",
            "scale_range": (0.5, 1.8),
            "rotation_range": 20,
        },
    ]

    # Metadata for HTML viewer
    metadata = {"created": datetime.now().isoformat(), "samples": []}

    print(f"Generating {num_samples} transformed images with precise shapes...")

    sample_idx = 0
    for i in range(num_samples):
        # Select configuration and background
        config = configs[i % len(configs)]
        bg_name = list(backgrounds.keys())[i % len(backgrounds)]
        bg_image = backgrounds[bg_name].copy()

        # Create transform
        transform = SimpleCopyPasteSegmented(
            coco_file=str(Path(base_dir) / "annotations.json"),
            object_counts=config["object_counts"],
            blend_mode=config["blend_mode"],
            scale_range=config["scale_range"],
            rotation_range=config["rotation_range"],
            p=1.0,
        )

        # Save original
        orig_filename = f"sample_{sample_idx:03d}_original_{bg_name}.jpg"
        cv2.imwrite(str(images_dir / orig_filename), bg_image)

        # Generate 3 variations
        for var in range(3):
            start_time = time.time()
            augmented = transform(image=bg_image)["image"]
            transform_time = (time.time() - start_time) * 1000

            trans_filename = (
                f"sample_{sample_idx:03d}_transformed_{bg_name}_v{var+1}.jpg"
            )
            cv2.imwrite(str(images_dir / trans_filename), augmented)

            # Add to metadata
            metadata["samples"].append(
                {
                    "index": sample_idx,
                    "original": orig_filename,
                    "transformed": trans_filename,
                    "background": bg_name,
                    "config": config["name"],
                    "object_counts": config["object_counts"],
                    "blend_mode": config["blend_mode"],
                    "scale_range": config["scale_range"],
                    "rotation_range": config["rotation_range"],
                    "transform_time_ms": transform_time,
                    "variation": var + 1,
                }
            )

        sample_idx += 1

        if (i + 1) % 5 == 0:
            print(f"  Generated {i + 1}/{num_samples} images...")

    # Create overview grids
    print("\nCreating overview grids...")

    # Transformation grid
    grid_rows = []
    for config in configs[:4]:  # Show first 4 configs
        row_images = []

        # Config label
        label_img = np.ones((244, 324, 3), dtype=np.uint8) * 255
        cv2.putText(
            label_img,
            config["name"].replace("_", " ").title(),
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2,
        )
        cv2.putText(
            label_img,
            f"Rotation: ±{config['rotation_range']}°",
            (10, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (100, 100, 100),
            1,
        )
        cv2.putText(
            label_img,
            f"Scale: {config['scale_range']}",
            (10, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (100, 100, 100),
            1,
        )
        objects_text = ", ".join(
            [f"{k}: {v}" for k, v in list(config["object_counts"].items())[:2]]
        )
        cv2.putText(
            label_img,
            objects_text,
            (10, 160),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (100, 100, 100),
            1,
        )
        row_images.append(label_img)

        # Create transform
        transform = SimpleCopyPasteSegmented(
            coco_file=str(Path(base_dir) / "annotations.json"),
            object_counts=config["object_counts"],
            blend_mode=config["blend_mode"],
            scale_range=config["scale_range"],
            rotation_range=config["rotation_range"],
            p=1.0,
        )

        # Apply to different backgrounds
        for bg_name in ["white", "gradient", "pattern"]:
            augmented = transform(image=backgrounds[bg_name])["image"]
            resized = cv2.resize(augmented, (320, 240))
            bordered = cv2.copyMakeBorder(
                resized, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(200, 200, 200)
            )
            row_images.append(bordered)

        row = np.hstack(row_images)
        grid_rows.append(row)

    grid = np.vstack(grid_rows)

    # Add header
    header = np.ones((60, grid.shape[1], 3), dtype=np.uint8) * 240
    cv2.putText(
        header,
        "FastSCP with Precise Segmentation Masks",
        (400, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 0, 0),
        2,
    )

    # Add column labels
    col_labels = np.ones((40, grid.shape[1], 3), dtype=np.uint8) * 250
    cv2.putText(
        col_labels,
        "Configuration",
        (80, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 0),
        1,
    )
    cv2.putText(
        col_labels, "White BG", (420, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1
    )
    cv2.putText(
        col_labels,
        "Gradient BG",
        (720, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 0),
        1,
    )
    cv2.putText(
        col_labels,
        "Pattern BG",
        (1040, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 0),
        1,
    )

    full_grid = np.vstack([header, col_labels, grid])
    cv2.imwrite(str(output_path / "transformation_grid.jpg"), full_grid)

    # Create before/after comparisons
    print("Creating before/after comparisons...")
    comparisons = []

    for i in range(2):  # 2 sets of comparisons
        for bg_name in ["gradient", "pattern"]:
            orig = backgrounds[bg_name]

            # Apply different configs
            results = [cv2.resize(orig, (200, 150))]

            for j in range(3):
                config = configs[j]
                transform = SimpleCopyPasteSegmented(
                    coco_file=str(Path(base_dir) / "annotations.json"),
                    object_counts=config["object_counts"],
                    blend_mode="alpha",
                    scale_range=config["scale_range"],
                    rotation_range=config["rotation_range"],
                    p=1.0,
                )
                augmented = transform(image=orig)["image"]
                results.append(cv2.resize(augmented, (200, 150)))

            comparison = np.hstack(results)

            # Add labels
            label_strip = np.ones((30, comparison.shape[1], 3), dtype=np.uint8) * 230
            cv2.putText(
                label_strip,
                f"{bg_name} background",
                (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                1,
            )
            cv2.putText(
                label_strip,
                "Original",
                (20, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
            )

            comparison_with_label = np.vstack([label_strip, comparison])
            comparisons.append(comparison_with_label)

    if comparisons:
        all_comparisons = np.vstack(comparisons)
        cv2.imwrite(str(output_path / "before_after_comparisons.jpg"), all_comparisons)

    # Save metadata
    with open(output_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("\nDataset generated:")
    print(f"  - {len(metadata['samples'])} augmented samples")
    print(f"  - Saved to: {output_path}")

    # Print performance summary
    times = [s["transform_time_ms"] for s in metadata["samples"]]
    print("\nPerformance Summary:")
    print(f"  Average transform time: {np.mean(times):.2f} ms")
    print(f"  Min/Max: {np.min(times):.2f} / {np.max(times):.2f} ms")

    return str(output_path)


def create_html_viewer(
    metadata_file="precise_transformation_samples/metadata.json",
    output_file="precise_transformation_viewer.html",
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

    # Calculate stats
    avg_time = sum(s["transform_time_ms"] for s in metadata["samples"]) / len(
        metadata["samples"]
    )
    total_samples = len(metadata["samples"])
    created_date = metadata["created"]

    # Generate HTML
    html_parts = []

    # Header
    html_parts.append(
        """<!DOCTYPE html>
<html>
<head>
    <title>FastSCP Precise Shapes Transformation Viewer</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
        h1 { color: #333; }
        .overview { margin: 20px 0; text-align: center; }
        .overview img { max-width: 100%; border: 2px solid #ddd; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
        .sample-group { background: white; margin: 20px 0; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .sample-header { font-size: 18px; font-weight: bold; margin-bottom: 10px; color: #444; }
        .metadata { background: #f8f8f8; padding: 10px; border-radius: 4px; margin-bottom: 15px; font-size: 14px; }
        .images { display: flex; gap: 15px; flex-wrap: wrap; align-items: center; }
        .image-container { text-align: center; }
        .image-container img { max-width: 300px; border: 1px solid #ddd; border-radius: 4px; }
        .image-label { margin-top: 5px; font-size: 12px; color: #666; }
        .performance { color: #4CAF50; font-weight: bold; }
        .stats { background: #e8f5e9; padding: 15px; border-radius: 8px; margin: 20px 0; }
        .feature { background: #e3f2fd; padding: 10px; margin: 5px 0; border-radius: 4px; }
    </style>
</head>
<body>
    <h1>FastSCP Precise Shapes Transformation Viewer</h1>

    <div class="feature">
        <h3>✨ Using Precise Segmentation Masks</h3>
        <p>This visualization shows transformations using exact shape boundaries (not rectangular bounding boxes).
        Notice the clean edges, rotation support, and lack of background artifacts!</p>
    </div>

    <div class="overview">
        <h2>Transformation Overview</h2>
        <img src="precise_transformation_samples/transformation_grid.jpg" alt="Transformation Grid">

        <h2>Before/After Comparisons</h2>
        <img src="precise_transformation_samples/before_after_comparisons.jpg" alt="Before/After Comparisons">
    </div>

    <div class="stats">
        <h3>Performance Statistics</h3>"""
    )

    html_parts.append(
        f'        <p>Average transformation time: <span class="performance">{avg_time:.2f} ms</span></p>'
    )
    html_parts.append(f"        <p>Total samples generated: {total_samples}</p>")
    html_parts.append(f"        <p>Created: {created_date}</p>")

    html_parts.append(
        """    </div>

    <h2>Individual Samples</h2>"""
    )

    # Add each sample group
    for orig_file, group in groups.items():
        config_name = group["config"].replace("_", " ").title()
        html_parts.append(
            f"""
    <div class="sample-group">
        <div class="sample-header">{config_name} - {group["background"]} background</div>
        <div class="metadata">"""
        )

        # Add metadata from first transform
        if group["transforms"]:
            t = group["transforms"][0]
            obj_str = ", ".join(f"{k}: {v}" for k, v in t["object_counts"].items())
            html_parts.append(f"            <strong>Objects:</strong> {obj_str}<br>")
            html_parts.append(
                f'            <strong>Rotation:</strong> ±{t["rotation_range"]}°<br>'
            )
            html_parts.append(
                f'            <strong>Scale Range:</strong> {t["scale_range"]}<br>'
            )
            html_parts.append(
                f'            <strong>Blend Mode:</strong> {t["blend_mode"]} (smooth edges)<br>'
            )

        html_parts.append(
            """        </div>
        <div class="images">
            <div class="image-container">"""
        )
        html_parts.append(
            f'                <img src="precise_transformation_samples/images/{orig_file}" alt="Original">'
        )
        html_parts.append(
            """                <div class="image-label">Original</div>
            </div>"""
        )

        # Add transformed versions
        for t in group["transforms"]:
            html_parts.append(
                """
            <div class="image-container">"""
            )
            html_parts.append(
                f'                <img src="precise_transformation_samples/images/{t["transformed"]}" alt="Transformed">'
            )
            html_parts.append(
                f'                <div class="image-label">Variant {t["variation"]} ({t["transform_time_ms"]:.2f} ms)</div>'
            )
            html_parts.append("""            </div>""")

        html_parts.append(
            """
        </div>
    </div>"""
        )

    html_parts.append(
        """
</body>
</html>"""
    )

    # Join all parts
    html = "\n".join(html_parts)

    # Save HTML
    with open(output_file, "w") as f:
        f.write(html)

    print(f"\nHTML viewer created: {output_file}")
    print("Open this file in your browser to view the transformations!")

    return output_file


def main():
    """Main function."""
    print("FastSCP Precise Shapes Visualization")
    print("=" * 50)

    # Generate transformed dataset
    dataset_dir = generate_transformed_dataset(num_samples=15)

    # Create HTML viewer
    output_file = create_html_viewer()

    # Try to open in browser
    import webbrowser

    file_path = os.path.abspath(output_file)
    url = f"file://{file_path}"
    print(f"\nOpening in browser: {url}")
    webbrowser.open(url)


if __name__ == "__main__":
    main()
