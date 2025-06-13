"""Simple visualization of transformations without FiftyOne."""

import json
import os
from pathlib import Path
from datetime import datetime
import time

import cv2
import numpy as np

from fastscp import SimpleCopyPaste


def generate_transformed_samples(base_dir="tests/colored_shapes", output_dir="transformation_samples", num_samples=12):
    """Generate a grid of transformed samples."""
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Load base images
    base_images = {
        "gray": cv2.imread(str(Path(base_dir) / "images" / "background_gray.jpg")),
        "gradient": cv2.imread(str(Path(base_dir) / "images" / "background_gradient.jpg")),
        "grid": cv2.imread(str(Path(base_dir) / "images" / "background_grid.jpg"))
    }
    
    # Different transformation configurations
    configs = [
        {
            "name": "Basic (2 red, 1 green)",
            "object_counts": {"red_square": 2, "green_circle": 1},
            "blend_mode": "overlay",
            "scale_range": (0.8, 1.2)
        },
        {
            "name": "Mix blend (triangles)",
            "object_counts": {"blue_triangle": 3, "yellow_rectangle": 1},
            "blend_mode": "mix",
            "scale_range": (0.5, 1.5)
        },
        {
            "name": "Many small shapes",
            "object_counts": {"red_square": 4, "green_circle": 3},
            "blend_mode": "overlay",
            "scale_range": (0.3, 0.6)
        },
        {
            "name": "Few large shapes",
            "object_counts": {"purple_star": 2, "cyan_hexagon": 1},
            "blend_mode": "overlay",
            "scale_range": (1.5, 2.5)
        }
    ]
    
    print("Generating transformed samples...")
    
    # Create grid of samples
    grid_rows = []
    
    for config in configs:
        row_images = []
        
        # Add config label (same height as transformed images)
        label_img = np.ones((244, 324, 3), dtype=np.uint8) * 255
        cv2.putText(label_img, config["name"], (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cv2.putText(label_img, f"Scale: {config['scale_range']}", (10, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
        cv2.putText(label_img, f"Blend: {config['blend_mode']}", (10, 130), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
        row_images.append(label_img)
        
        # Create transform
        transform = SimpleCopyPaste(
            coco_file=str(Path(base_dir) / "annotations.json"),
            object_counts=config["object_counts"],
            blend_mode=config["blend_mode"],
            scale_range=config["scale_range"],
            p=1.0
        )
        
        # Apply to each background
        for bg_name, bg_img in base_images.items():
            # Apply transformation
            augmented = transform(image=bg_img)["image"]
            
            # Resize for grid
            resized = cv2.resize(augmented, (320, 240))
            
            # Add border
            bordered = cv2.copyMakeBorder(resized, 2, 2, 2, 2, 
                                        cv2.BORDER_CONSTANT, value=(200, 200, 200))
            
            row_images.append(bordered)
        
        # Combine row
        row = np.hstack(row_images)
        grid_rows.append(row)
    
    # Create full grid
    grid = np.vstack(grid_rows)
    
    # Add header
    header = np.ones((60, grid.shape[1], 3), dtype=np.uint8) * 240
    cv2.putText(header, "FastSCP Transformation Examples", (20, 40), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
    
    # Add column labels
    col_labels = np.ones((40, grid.shape[1], 3), dtype=np.uint8) * 250
    cv2.putText(col_labels, "Configuration", (50, 25), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    cv2.putText(col_labels, "Gray BG", (390, 25), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    cv2.putText(col_labels, "Gradient BG", (690, 25), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    cv2.putText(col_labels, "Grid BG", (1020, 25), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    
    # Combine all
    full_image = np.vstack([header, col_labels, grid])
    
    # Save
    output_file = output_path / "transformation_grid.jpg"
    cv2.imwrite(str(output_file), full_image)
    print(f"\nSaved transformation grid to: {output_file}")
    
    # Generate individual samples with metadata
    print("\nGenerating individual samples with metadata...")
    
    metadata = {
        "created": datetime.now().isoformat(),
        "samples": []
    }
    
    sample_idx = 0
    for config in configs[:2]:  # Just first 2 configs for individual samples
        transform = SimpleCopyPaste(
            coco_file=str(Path(base_dir) / "annotations.json"),
            object_counts=config["object_counts"],
            blend_mode=config["blend_mode"],
            scale_range=config["scale_range"],
            p=1.0
        )
        
        for bg_name, bg_img in base_images.items():
            # Original
            orig_file = f"sample_{sample_idx:03d}_original_{bg_name}.jpg"
            cv2.imwrite(str(output_path / orig_file), bg_img)
            
            # Transformed (3 variations)
            for var in range(3):
                start_time = time.time()
                augmented = transform(image=bg_img)["image"]
                transform_time = (time.time() - start_time) * 1000
                
                trans_file = f"sample_{sample_idx:03d}_transformed_{bg_name}_v{var+1}.jpg"
                cv2.imwrite(str(output_path / trans_file), augmented)
                
                # Add to metadata
                metadata["samples"].append({
                    "index": sample_idx,
                    "original": orig_file,
                    "transformed": trans_file,
                    "background": bg_name,
                    "config": config["name"],
                    "object_counts": config["object_counts"],
                    "blend_mode": config["blend_mode"],
                    "scale_range": config["scale_range"],
                    "transform_time_ms": transform_time,
                    "variation": var + 1
                })
            
            sample_idx += 1
    
    # Save metadata
    metadata_file = output_path / "metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Generated {len(metadata['samples'])} individual samples")
    print(f"Metadata saved to: {metadata_file}")
    
    # Create comparison strips
    print("\nCreating before/after comparison strips...")
    
    comparison_strips = []
    for i in range(0, min(6, len(metadata["samples"])), 3):
        sample = metadata["samples"][i]
        
        # Load images
        orig = cv2.imread(str(output_path / sample["original"]))
        trans1 = cv2.imread(str(output_path / metadata["samples"][i]["transformed"]))
        trans2 = cv2.imread(str(output_path / metadata["samples"][i+1]["transformed"]))
        trans3 = cv2.imread(str(output_path / metadata["samples"][i+2]["transformed"]))
        
        # Resize
        size = (200, 150)
        orig = cv2.resize(orig, size)
        trans1 = cv2.resize(trans1, size)
        trans2 = cv2.resize(trans2, size)
        trans3 = cv2.resize(trans3, size)
        
        # Create strip
        strip = np.hstack([orig, trans1, trans2, trans3])
        
        # Add labels
        label_height = 30
        label_strip = np.ones((label_height, strip.shape[1], 3), dtype=np.uint8) * 230
        cv2.putText(label_strip, "Original", (20, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.putText(label_strip, "Variant 1", (220, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.putText(label_strip, "Variant 2", (420, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.putText(label_strip, "Variant 3", (620, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Combine
        comparison = np.vstack([label_strip, strip])
        comparison_strips.append(comparison)
    
    # Save comparison image
    if comparison_strips:
        all_comparisons = np.vstack(comparison_strips)
        comparison_file = output_path / "before_after_comparisons.jpg"
        cv2.imwrite(str(comparison_file), all_comparisons)
        print(f"Saved comparisons to: {comparison_file}")
    
    print("\n" + "="*50)
    print("Visualization complete!")
    print(f"All outputs saved to: {output_path}/")
    
    # Print summary statistics
    if metadata["samples"]:
        times = [s["transform_time_ms"] for s in metadata["samples"]]
        print(f"\nPerformance Summary:")
        print(f"  Average transform time: {np.mean(times):.2f} ms")
        print(f"  Min/Max: {np.min(times):.2f} / {np.max(times):.2f} ms")


def main():
    """Main function."""
    print("FastSCP Transformation Visualization")
    print("=" * 50)
    
    generate_transformed_samples()


if __name__ == "__main__":
    main()