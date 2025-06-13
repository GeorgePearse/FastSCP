"""Test script for segmentation-based SimpleCopyPaste transformation."""

import cv2
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from fastscp.transforms_segmented import SimpleCopyPasteSegmented
from fastscp.coco_loader_segmented import COCOLoaderSegmented


def visualize_segmentation_crops():
    """Visualize how segmentation-based cropping works."""
    
    print("Testing segmentation-based cropping...")
    
    # Load COCO loader
    coco_file = "tests/segmented_shapes/annotations.json"
    loader = COCOLoaderSegmented(coco_file)
    
    # Create output directory
    output_dir = Path("segmentation_test_output")
    output_dir.mkdir(exist_ok=True)
    
    # Test loading individual shapes with masks
    print("\n1. Loading individual shapes with precise segmentation...")
    
    categories = ["square", "circle", "triangle", "star", "hexagon"]
    comparison_images = []
    
    for cat_name in categories:
        # Get one object from each category
        objects = loader.get_random_objects(cat_name, 1, include_context=0.0)
        
        if objects:
            obj_img, obj_mask = objects[0]
            
            # Create visualization
            vis_img = np.ones((200, 600, 3), dtype=np.uint8) * 255
            
            # Original object (remove alpha for display)
            if obj_img.shape[2] == 4:
                obj_rgb = obj_img[:, :, :3]
                obj_alpha = obj_img[:, :, 3]
            else:
                obj_rgb = obj_img
                obj_alpha = obj_mask
            
            # Resize to fit
            h, w = obj_rgb.shape[:2]
            scale = min(180 / h, 180 / w)
            new_h, new_w = int(h * scale), int(w * scale)
            obj_resized = cv2.resize(obj_rgb, (new_w, new_h))
            mask_resized = cv2.resize(obj_alpha, (new_w, new_h))
            
            # Place in visualization
            y_offset = (200 - new_h) // 2
            x_offset = 10
            
            # Original with background
            vis_img[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = obj_resized
            
            # Mask visualization
            mask_vis = cv2.cvtColor(mask_resized, cv2.COLOR_GRAY2BGR)
            x_offset += 200
            vis_img[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = mask_vis
            
            # Object with transparent background (checkerboard pattern)
            x_offset += 200
            checker = np.zeros((new_h, new_w, 3), dtype=np.uint8)
            for i in range(0, new_h, 10):
                for j in range(0, new_w, 10):
                    if (i // 10 + j // 10) % 2 == 0:
                        checker[i:i+10, j:j+10] = 200
                    else:
                        checker[i:i+10, j:j+10] = 150
            
            # Apply mask
            mask_norm = mask_resized.astype(np.float32) / 255.0
            if len(mask_norm.shape) == 2:
                mask_norm = mask_norm[:, :, np.newaxis]
            
            masked_obj = (obj_resized * mask_norm + checker * (1 - mask_norm)).astype(np.uint8)
            vis_img[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = masked_obj
            
            # Add labels
            cv2.putText(vis_img, f"{cat_name}", (10, 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            cv2.putText(vis_img, "Original", (10, 190), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
            cv2.putText(vis_img, "Mask", (210, 190), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
            cv2.putText(vis_img, "Masked", (410, 190), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
            
            comparison_images.append(vis_img)
    
    # Save comparison
    if comparison_images:
        comparison = np.vstack(comparison_images)
        cv2.imwrite(str(output_dir / "segmentation_crops_comparison.jpg"), comparison)
        print(f"   Saved: segmentation_crops_comparison.jpg")
    
    # Test the transform
    print("\n2. Testing SimpleCopyPasteSegmented transform...")
    
    # Create test backgrounds
    backgrounds = {
        "white": np.ones((600, 800, 3), dtype=np.uint8) * 255,
        "gray": np.ones((600, 800, 3), dtype=np.uint8) * 180,
        "gradient": np.zeros((600, 800, 3), dtype=np.uint8)
    }
    
    # Create gradient
    for i in range(600):
        backgrounds["gradient"][i, :] = int(255 * (i / 600))
    
    # Test different configurations
    configs = [
        {
            "name": "Basic shapes (no rotation)",
            "object_counts": {"square": 2, "circle": 2, "triangle": 1},
            "rotation_range": 0,
            "scale_range": (0.8, 1.2),
            "blend_mode": "alpha"
        },
        {
            "name": "With rotation",
            "object_counts": {"star": 2, "hexagon": 2, "pentagon": 1},
            "rotation_range": 45,
            "scale_range": (0.5, 1.5),
            "blend_mode": "alpha"
        },
        {
            "name": "Many small shapes",
            "object_counts": {"square": 3, "circle": 3, "triangle": 2},
            "rotation_range": 30,
            "scale_range": (0.3, 0.6),
            "blend_mode": "alpha"
        },
        {
            "name": "Mix blend mode",
            "object_counts": {"star": 2, "diamond": 1, "octagon": 1},
            "rotation_range": 15,
            "scale_range": (0.7, 1.3),
            "blend_mode": "mix"
        }
    ]
    
    # Generate transformations
    for config in configs:
        print(f"\n   Testing: {config['name']}")
        
        transform = SimpleCopyPasteSegmented(
            coco_file=coco_file,
            object_counts=config["object_counts"],
            rotation_range=config["rotation_range"],
            scale_range=config["scale_range"],
            blend_mode=config["blend_mode"],
            p=1.0
        )
        
        # Create grid showing results on different backgrounds
        grid_rows = []
        
        for bg_name, bg_img in backgrounds.items():
            row_images = [cv2.resize(bg_img, (200, 150))]  # Original
            
            # Generate 3 variations
            for i in range(3):
                augmented = transform(image=bg_img)["image"]
                row_images.append(cv2.resize(augmented, (200, 150)))
            
            # Stack horizontally
            row = np.hstack(row_images)
            
            # Add background label
            label_img = np.ones((30, row.shape[1], 3), dtype=np.uint8) * 240
            cv2.putText(label_img, f"{bg_name} background", (10, 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            
            row_with_label = np.vstack([label_img, row])
            grid_rows.append(row_with_label)
        
        # Create full grid
        grid = np.vstack(grid_rows)
        
        # Add header
        header = np.ones((60, grid.shape[1], 3), dtype=np.uint8) * 230
        cv2.putText(header, config['name'], (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.putText(header, f"Rotation: ±{config['rotation_range']}°, "
                           f"Scale: {config['scale_range']}, "
                           f"Blend: {config['blend_mode']}", 
                   (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80), 1)
        
        # Add column labels
        col_labels = np.ones((25, grid.shape[1], 3), dtype=np.uint8) * 250
        cv2.putText(col_labels, "Original", (60, 18), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.putText(col_labels, "Variation 1", (250, 18), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.putText(col_labels, "Variation 2", (450, 18), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.putText(col_labels, "Variation 3", (650, 18), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Combine all
        full_grid = np.vstack([header, col_labels, grid])
        
        # Save
        filename = f"transform_{config['name'].lower().replace(' ', '_').replace('(', '').replace(')', '')}.jpg"
        cv2.imwrite(str(output_dir / filename), full_grid)
        print(f"   Saved: {filename}")
    
    # Create side-by-side comparison of bbox vs segmentation
    print("\n3. Creating bbox vs segmentation comparison...")
    
    # Load both transforms
    from fastscp.transforms import SimpleCopyPaste
    
    transform_bbox = SimpleCopyPaste(
        coco_file="tests/colored_shapes/annotations.json",
        object_counts={"red_square": 2, "green_circle": 2},
        p=1.0
    )
    
    transform_seg = SimpleCopyPasteSegmented(
        coco_file=coco_file,
        object_counts={"square": 2, "circle": 2},
        blend_mode="alpha",
        p=1.0
    )
    
    # Create comparison
    bg = backgrounds["gradient"].copy()
    
    # Apply both transforms
    result_bbox = transform_bbox(image=bg)["image"]
    result_seg = transform_seg(image=bg)["image"]
    
    # Create comparison image
    comparison = np.hstack([
        cv2.resize(bg, (400, 300)),
        cv2.resize(result_bbox, (400, 300)),
        cv2.resize(result_seg, (400, 300))
    ])
    
    # Add labels
    label_height = 40
    labels = np.ones((label_height, comparison.shape[1], 3), dtype=np.uint8) * 240
    cv2.putText(labels, "Original", (150, 28), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(labels, "BBox-based Crop", (480, 28), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(labels, "Segmentation-based Crop", (850, 28), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    
    final_comparison = np.vstack([labels, comparison])
    cv2.imwrite(str(output_dir / "bbox_vs_segmentation_comparison.jpg"), final_comparison)
    print("   Saved: bbox_vs_segmentation_comparison.jpg")
    
    print(f"\nAll outputs saved to: {output_dir}/")
    print("\nSegmentation-based cropping provides:")
    print("  - Precise object boundaries")
    print("  - Smooth alpha blending at edges")
    print("  - No background artifacts")
    print("  - Support for rotation without rectangular borders")


if __name__ == "__main__":
    visualize_segmentation_crops()