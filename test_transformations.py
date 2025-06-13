"""Test script to demonstrate SimpleCopyPaste transformations with colored shapes."""

import os
import time
from pathlib import Path

import cv2
import numpy as np
import albumentations as A

from fastscp import SimpleCopyPaste


def create_output_grid(images, titles, grid_size=(2, 3), img_size=(320, 240)):
    """Create a grid of images for visualization."""
    rows, cols = grid_size
    grid = np.ones((rows * img_size[1], cols * img_size[0], 3), dtype=np.uint8) * 255
    
    for idx, (img, title) in enumerate(zip(images, titles)):
        if idx >= rows * cols:
            break
            
        row = idx // cols
        col = idx % cols
        
        # Resize image to fit
        resized = cv2.resize(img, img_size)
        
        # Place in grid
        y1 = row * img_size[1]
        y2 = (row + 1) * img_size[1]
        x1 = col * img_size[0]
        x2 = (col + 1) * img_size[0]
        
        grid[y1:y2, x1:x2] = resized
        
        # Add title
        cv2.putText(grid, title, (x1 + 10, y1 + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    return grid


def main():
    """Run various transformations and save results."""
    
    # Setup paths
    data_dir = Path("tests/colored_shapes")
    output_dir = Path("output_transformations")
    output_dir.mkdir(exist_ok=True)
    
    # Load background images
    backgrounds = {
        "gray": cv2.imread(str(data_dir / "images" / "background_gray.jpg")),
        "gradient": cv2.imread(str(data_dir / "images" / "background_gradient.jpg")),
        "grid": cv2.imread(str(data_dir / "images" / "background_grid.jpg"))
    }
    
    print("FastSCP Transformation Tests")
    print("=" * 50)
    
    # Test 1: Basic overlay with different object counts
    print("\n1. Testing different object counts...")
    
    results = []
    titles = []
    
    for bg_name, bg_img in backgrounds.items():
        # Original
        results.append(bg_img.copy())
        titles.append(f"Original {bg_name}")
        
        # With 2 red squares and 1 green circle
        transform = SimpleCopyPaste(
            coco_file=str(data_dir / "annotations.json"),
            object_counts={"red_square": 2, "green_circle": 1},
            p=1.0
        )
        augmented = transform(image=bg_img)["image"]
        results.append(augmented)
        titles.append("2 red, 1 green")
    
    grid = create_output_grid(results, titles)
    cv2.imwrite(str(output_dir / "test1_object_counts.jpg"), grid)
    print("   ✓ Saved: test1_object_counts.jpg")
    
    # Test 2: Different blend modes
    print("\n2. Testing blend modes...")
    
    bg_img = backgrounds["gray"]
    results = [bg_img.copy()]
    titles = ["Original"]
    
    for blend_mode in ["overlay", "mix"]:
        transform = SimpleCopyPaste(
            coco_file=str(data_dir / "annotations.json"),
            object_counts={"blue_triangle": 2, "yellow_rectangle": 2},
            blend_mode=blend_mode,
            p=1.0
        )
        augmented = transform(image=bg_img)["image"]
        results.append(augmented)
        titles.append(f"Blend: {blend_mode}")
    
    grid = create_output_grid(results[:3], titles[:3], grid_size=(1, 3))
    cv2.imwrite(str(output_dir / "test2_blend_modes.jpg"), grid)
    print("   ✓ Saved: test2_blend_modes.jpg")
    
    # Test 3: Scale variations
    print("\n3. Testing scale variations...")
    
    bg_img = backgrounds["grid"]
    results = [bg_img.copy()]
    titles = ["Original"]
    
    for scale_range in [(0.3, 0.5), (0.8, 1.2), (1.5, 2.0)]:
        transform = SimpleCopyPaste(
            coco_file=str(data_dir / "annotations.json"),
            object_counts={"purple_star": 1, "cyan_hexagon": 1},
            scale_range=scale_range,
            p=1.0
        )
        augmented = transform(image=bg_img)["image"]
        results.append(augmented)
        titles.append(f"Scale: {scale_range}")
    
    grid = create_output_grid(results[:4], titles[:4], grid_size=(2, 2))
    cv2.imwrite(str(output_dir / "test3_scale_variations.jpg"), grid)
    print("   ✓ Saved: test3_scale_variations.jpg")
    
    # Test 4: Complex pipeline
    print("\n4. Testing complex augmentation pipeline...")
    
    pipeline = A.Compose([
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        SimpleCopyPaste(
            coco_file=str(data_dir / "annotations.json"),
            object_counts={
                "red_square": 1,
                "green_circle": 1,
                "blue_triangle": 1,
                "yellow_rectangle": 1
            },
            scale_range=(0.5, 1.5),
            blend_mode="mix",
            p=0.8
        ),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.3),
    ])
    
    bg_img = backgrounds["gradient"]
    results = [bg_img.copy()]
    titles = ["Original"]
    
    # Generate multiple augmentations
    for i in range(5):
        augmented = pipeline(image=bg_img)["image"]
        results.append(augmented)
        titles.append(f"Augmented {i+1}")
    
    grid = create_output_grid(results, titles)
    cv2.imwrite(str(output_dir / "test4_pipeline.jpg"), grid)
    print("   ✓ Saved: test4_pipeline.jpg")
    
    # Test 5: Performance with many objects
    print("\n5. Testing performance...")
    
    transform = SimpleCopyPaste(
        coco_file=str(data_dir / "annotations.json"),
        object_counts={
            "red_square": 5,
            "green_circle": 5,
            "blue_triangle": 5,
        },
        p=1.0
    )
    
    # Warmup
    bg_img = backgrounds["gray"]
    transform(image=bg_img)
    
    # Time multiple runs
    times = []
    for _ in range(50):
        start = time.time()
        transform(image=bg_img)
        times.append(time.time() - start)
    
    avg_time = np.mean(times) * 1000
    print(f"   Average time with 15 objects: {avg_time:.2f} ms")
    
    # Create one example with many objects
    augmented = transform(image=bg_img)["image"]
    cv2.imwrite(str(output_dir / "test5_many_objects.jpg"), augmented)
    print("   ✓ Saved: test5_many_objects.jpg")
    
    # Test 6: Edge cases
    print("\n6. Testing edge cases...")
    
    # Very small scale
    transform_small = SimpleCopyPaste(
        coco_file=str(data_dir / "annotations.json"),
        object_counts={"red_square": 10},
        scale_range=(0.1, 0.2),
        p=1.0
    )
    
    # Very large scale  
    transform_large = SimpleCopyPaste(
        coco_file=str(data_dir / "annotations.json"),
        object_counts={"green_circle": 3},
        scale_range=(2.0, 3.0),
        p=1.0
    )
    
    bg_img = backgrounds["grid"]
    results = [
        bg_img.copy(),
        transform_small(image=bg_img)["image"],
        transform_large(image=bg_img)["image"]
    ]
    titles = ["Original", "Very small objects", "Very large objects"]
    
    grid = create_output_grid(results, titles, grid_size=(1, 3))
    cv2.imwrite(str(output_dir / "test6_edge_cases.jpg"), grid)
    print("   ✓ Saved: test6_edge_cases.jpg")
    
    print("\n" + "=" * 50)
    print(f"All tests completed! Results saved to {output_dir}/")
    print("\nFiles created:")
    for file in sorted(output_dir.glob("*.jpg")):
        print(f"  - {file.name}")


if __name__ == "__main__":
    main()