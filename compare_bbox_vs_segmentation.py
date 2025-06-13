"""Create a clear comparison between bbox-based and segmentation-based cropping."""

import cv2
import numpy as np
from pathlib import Path

from fastscp.transforms import SimpleCopyPaste
from fastscp.transforms_segmented import SimpleCopyPasteSegmented


def create_comparison():
    """Create side-by-side comparison of bbox vs segmentation approaches."""
    
    print("Creating bbox vs segmentation comparison...")
    
    # Create a test background with gradient
    background = np.zeros((600, 800, 3), dtype=np.uint8)
    for i in range(600):
        background[i, :] = int(128 + 127 * (i / 600))
    
    # Add some features to make it interesting
    cv2.rectangle(background, (50, 50), (200, 150), (100, 150, 200), -1)
    cv2.circle(background, (700, 100), 60, (200, 150, 100), -1)
    cv2.putText(background, "Test Background", (300, 550), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Create transforms
    print("Creating bbox-based transform...")
    transform_bbox = SimpleCopyPaste(
        coco_file="tests/colored_shapes/annotations.json",
        object_counts={"red_square": 2, "green_circle": 2, "blue_triangle": 1},
        scale_range=(0.8, 1.5),
        blend_mode="overlay",
        p=1.0
    )
    
    print("Creating segmentation-based transform...")
    transform_seg = SimpleCopyPasteSegmented(
        coco_file="tests/precise_shapes/annotations.json",
        object_counts={"red_square": 2, "green_circle": 2, "blue_triangle": 1},
        scale_range=(0.8, 1.5),
        rotation_range=30,
        blend_mode="alpha",
        p=1.0
    )
    
    # Apply transforms
    print("Applying transformations...")
    result_bbox = transform_bbox(image=background.copy())["image"]
    result_seg = transform_seg(image=background.copy())["image"]
    
    # Create detailed comparison
    output_dir = Path("comparison_output")
    output_dir.mkdir(exist_ok=True)
    
    # Main comparison
    main_comparison = np.hstack([
        cv2.resize(background, (400, 300)),
        cv2.resize(result_bbox, (400, 300)),
        cv2.resize(result_seg, (400, 300))
    ])
    
    # Add labels
    label_height = 50
    labels = np.ones((label_height, main_comparison.shape[1], 3), dtype=np.uint8) * 50
    cv2.putText(labels, "Original", (150, 35), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(labels, "BBox-based (Rectangular)", (430, 35), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(labels, "Segmentation-based (Precise)", (820, 35), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    final_comparison = np.vstack([labels, main_comparison])
    cv2.imwrite(str(output_dir / "main_comparison.jpg"), final_comparison)
    print("Saved: main_comparison.jpg")
    
    # Create zoomed-in detail views
    print("\nCreating detail views...")
    
    # Find regions with shapes for close-up
    detail_views = []
    
    for i in range(3):
        # Apply transforms again for variety
        bbox_result = transform_bbox(image=background.copy())["image"]
        seg_result = transform_seg(image=background.copy())["image"]
        
        # Pick a random region
        x = np.random.randint(100, 500)
        y = np.random.randint(100, 300)
        size = 200
        
        # Extract regions
        region_orig = background[y:y+size, x:x+size]
        region_bbox = bbox_result[y:y+size, x:x+size]
        region_seg = seg_result[y:y+size, x:x+size]
        
        # Scale up for detail
        scale = 2
        region_orig_scaled = cv2.resize(region_orig, (size*scale, size*scale), interpolation=cv2.INTER_NEAREST)
        region_bbox_scaled = cv2.resize(region_bbox, (size*scale, size*scale), interpolation=cv2.INTER_NEAREST)
        region_seg_scaled = cv2.resize(region_seg, (size*scale, size*scale), interpolation=cv2.INTER_NEAREST)
        
        # Create detail comparison
        detail = np.hstack([region_orig_scaled, region_bbox_scaled, region_seg_scaled])
        
        # Add grid to show pixels
        for i in range(0, detail.shape[0], 20):
            cv2.line(detail, (0, i), (detail.shape[1], i), (100, 100, 100), 1)
        for i in range(0, detail.shape[1], 20):
            cv2.line(detail, (i, 0), (i, detail.shape[0]), (100, 100, 100), 1)
        
        detail_views.append(detail)
    
    if detail_views:
        all_details = np.vstack(detail_views)
        
        # Add header
        header = np.ones((40, all_details.shape[1], 3), dtype=np.uint8) * 50
        cv2.putText(header, "Original", (150, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(header, "BBox (Rectangle artifacts)", (450, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(header, "Segmentation (Clean edges)", (900, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        final_details = np.vstack([header, all_details])
        cv2.imwrite(str(output_dir / "detail_comparison.jpg"), final_details)
        print("Saved: detail_comparison.jpg")
    
    # Create summary image showing the key differences
    print("\nCreating summary...")
    
    summary = np.ones((800, 1200, 3), dtype=np.uint8) * 240
    
    # Title
    cv2.putText(summary, "BBox vs Segmentation-based Copy-Paste", (200, 50), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
    
    # BBox column
    cv2.putText(summary, "Bounding Box Method:", (50, 120), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    y_pos = 160
    issues = [
        "- Includes background pixels",
        "- Rectangular crops only",
        "- No rotation support",
        "- Hard edges",
        "- Artifacts at boundaries"
    ]
    for issue in issues:
        cv2.putText(summary, issue, (70, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 200), 2)
        y_pos += 35
    
    # Segmentation column
    cv2.putText(summary, "Segmentation Method:", (650, 120), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    y_pos = 160
    benefits = [
        "- Exact shape boundaries",
        "- Any shape supported",
        "- Clean rotation",
        "- Smooth anti-aliased edges",
        "- No background artifacts"
    ]
    for benefit in benefits:
        cv2.putText(summary, benefit, (670, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 150, 0), 2)
        y_pos += 35
    
    # Add example images
    example_bbox = cv2.resize(result_bbox[200:400, 300:500], (300, 300))
    example_seg = cv2.resize(result_seg[200:400, 300:500], (300, 300))
    
    summary[400:700, 100:400] = example_bbox
    summary[400:700, 700:1000] = example_seg
    
    cv2.putText(summary, "BBox Result", (200, 730), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(summary, "Segmentation Result", (770, 730), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    
    cv2.imwrite(str(output_dir / "method_comparison_summary.jpg"), summary)
    print("Saved: method_comparison_summary.jpg")
    
    print(f"\nComparison complete! All outputs saved to: {output_dir}/")
    print("\nKey differences:")
    print("1. BBox method: Crops rectangular regions, includes unwanted background")
    print("2. Segmentation method: Crops exact shape boundaries, clean edges")
    print("3. Segmentation supports rotation without artifacts")
    print("4. Segmentation provides smooth alpha blending")


if __name__ == "__main__":
    create_comparison()