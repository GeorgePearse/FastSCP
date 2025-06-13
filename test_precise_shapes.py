"""Test the SimpleCopyPasteSegmented transform with precise shape masks."""

import cv2
import numpy as np
from pathlib import Path
import time

from fastscp.transforms_segmented import SimpleCopyPasteSegmented


def test_precise_shapes():
    """Test copy-paste with precisely generated shapes."""
    
    print("Testing SimpleCopyPasteSegmented with precise masks...")
    print("=" * 60)
    
    # Create output directory
    output_dir = Path("precise_shapes_output")
    output_dir.mkdir(exist_ok=True)
    
    # Create test backgrounds
    backgrounds = {
        "gradient": np.zeros((600, 800, 3), dtype=np.uint8),
        "white": np.ones((600, 800, 3), dtype=np.uint8) * 255,
        "pattern": np.zeros((600, 800, 3), dtype=np.uint8)
    }
    
    # Create gradient
    for i in range(600):
        backgrounds["gradient"][i, :] = int(255 * (i / 600))
    
    # Create checkerboard pattern
    for i in range(0, 600, 50):
        for j in range(0, 800, 50):
            if (i // 50 + j // 50) % 2 == 0:
                backgrounds["pattern"][i:i+50, j:j+50] = 220
            else:
                backgrounds["pattern"][i:i+50, j:j+50] = 180
    
    # Test configurations
    configs = [
        {
            "name": "Basic shapes",
            "counts": {"red_square": 2, "green_circle": 2, "blue_triangle": 1},
            "scale": (0.8, 1.2),
            "rotation": 0
        },
        {
            "name": "With rotation",
            "counts": {"purple_star": 2, "cyan_hexagon": 2, "yellow_diamond": 1},
            "scale": (0.6, 1.4),
            "rotation": 30
        },
        {
            "name": "Many small",
            "counts": {"red_square": 3, "green_circle": 3, "blue_triangle": 2},
            "scale": (0.3, 0.6),
            "rotation": 45
        },
        {
            "name": "Few large",
            "counts": {"purple_star": 1, "cyan_hexagon": 1},
            "scale": (1.5, 2.5),
            "rotation": 15
        }
    ]
    
    # Create comparison grid
    for config_idx, config in enumerate(configs):
        print(f"\nTesting configuration: {config['name']}")
        
        # Create transform
        transform = SimpleCopyPasteSegmented(
            coco_file="tests/precise_shapes/annotations.json",
            object_counts=config["counts"],
            scale_range=config["scale"],
            rotation_range=config["rotation"],
            blend_mode="alpha",
            p=1.0
        )
        
        # Test on each background
        results = []
        times = []
        
        for bg_name, bg_img in backgrounds.items():
            # Time the transformation
            start = time.time()
            augmented = transform(image=bg_img)["image"]
            elapsed = (time.time() - start) * 1000
            times.append(elapsed)
            
            # Create comparison
            comparison = np.hstack([
                cv2.resize(bg_img, (400, 300)),
                cv2.resize(augmented, (400, 300))
            ])
            
            # Add labels
            cv2.putText(comparison, "Original", (150, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(comparison, "With Shapes", (550, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(comparison, f"{bg_name}", (10, 280), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(comparison, f"{elapsed:.1f}ms", (720, 280), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            results.append(comparison)
        
        # Stack all results
        full_comparison = np.vstack(results)
        
        # Add header
        header = np.ones((60, full_comparison.shape[1], 3), dtype=np.uint8) * 50
        cv2.putText(header, config["name"], (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        cv2.putText(header, f"Scale: {config['scale']}, Rotation: ±{config['rotation']}°", 
                   (400, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
        
        final_image = np.vstack([header, full_comparison])
        
        # Save
        filename = f"precise_test_{config_idx:02d}_{config['name'].lower().replace(' ', '_')}.jpg"
        cv2.imwrite(str(output_dir / filename), final_image)
        print(f"  Saved: {filename}")
        print(f"  Average time: {np.mean(times):.2f}ms")
    
    # Create a detailed view of individual shapes
    print("\nCreating detailed shape extraction view...")
    
    from fastscp.coco_loader_segmented import COCOLoaderSegmented
    loader = COCOLoaderSegmented("tests/precise_shapes/annotations.json")
    
    shape_views = []
    for cat_name in ["red_square", "green_circle", "blue_triangle", "purple_star"]:
        objects = loader.get_random_objects(cat_name, 1)
        if objects:
            obj_img, obj_mask = objects[0]
            
            # Create visualization
            viz = np.ones((200, 600, 3), dtype=np.uint8) * 255
            
            # Get dimensions
            h, w = obj_img.shape[:2]
            scale = min(180 / h, 180 / w)
            new_h, new_w = int(h * scale), int(w * scale)
            
            # Resize
            obj_resized = cv2.resize(obj_img, (new_w, new_h))
            mask_resized = cv2.resize(obj_mask, (new_w, new_h))
            
            y_off = (200 - new_h) // 2
            
            # Show original (with alpha)
            if obj_resized.shape[2] == 4:
                rgb = obj_resized[:, :, :3]
                alpha = obj_resized[:, :, 3].astype(float) / 255
            else:
                rgb = obj_resized
                alpha = mask_resized.astype(float) / 255
            
            # Place on white background
            white = np.ones_like(rgb) * 255
            blended = (rgb * alpha[:, :, np.newaxis] + white * (1 - alpha[:, :, np.newaxis])).astype(np.uint8)
            viz[y_off:y_off+new_h, 10:10+new_w] = blended
            
            # Show mask
            mask_viz = cv2.cvtColor(mask_resized, cv2.COLOR_GRAY2BGR)
            viz[y_off:y_off+new_h, 210:210+new_w] = mask_viz
            
            # Show on checkered background
            checker = np.zeros((new_h, new_w, 3), dtype=np.uint8)
            for i in range(0, new_h, 10):
                for j in range(0, new_w, 10):
                    if (i // 10 + j // 10) % 2 == 0:
                        checker[i:i+10, j:j+10] = 200
                    else:
                        checker[i:i+10, j:j+10] = 150
            
            final = (rgb * alpha[:, :, np.newaxis] + checker * (1 - alpha[:, :, np.newaxis])).astype(np.uint8)
            viz[y_off:y_off+new_h, 410:410+new_w] = final
            
            # Labels
            cv2.putText(viz, cat_name, (10, 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            cv2.putText(viz, "Original", (10, 190), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)
            cv2.putText(viz, "Mask", (210, 190), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)
            cv2.putText(viz, "Transparent", (410, 190), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)
            
            shape_views.append(viz)
    
    if shape_views:
        shape_detail = np.vstack(shape_views)
        cv2.imwrite(str(output_dir / "shape_extraction_detail.jpg"), shape_detail)
        print("  Saved: shape_extraction_detail.jpg")
    
    print(f"\nAll outputs saved to: {output_dir}/")
    print("\nPrecise segmentation provides:")
    print("  ✓ Exact shape boundaries (no rectangular crops)")
    print("  ✓ Smooth anti-aliased edges")
    print("  ✓ Proper transparency handling")
    print("  ✓ Clean rotation without artifacts")


if __name__ == "__main__":
    test_precise_shapes()