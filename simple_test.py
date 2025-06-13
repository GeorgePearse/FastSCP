"""Simple test script showing basic copy-paste transformation."""

import cv2
import numpy as np
from fastscp import SimpleCopyPaste


def main():
    """Run a simple transformation and save before/after comparison."""
    
    # Create a simple test image
    print("Creating test image...")
    image = np.ones((480, 640, 3), dtype=np.uint8) * 220  # Light gray
    
    # Add some features to make it interesting
    cv2.rectangle(image, (50, 50), (200, 150), (180, 180, 180), -1)
    cv2.circle(image, (500, 350), 60, (200, 200, 200), -1)
    cv2.putText(image, "Original Image", (220, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)
    
    # Create transform
    print("Setting up SimpleCopyPaste transform...")
    transform = SimpleCopyPaste(
        coco_file="tests/colored_shapes/annotations.json",
        object_counts={
            "red_square": 2,
            "green_circle": 2,
            "blue_triangle": 1,
            "yellow_rectangle": 1
        },
        scale_range=(0.5, 1.2),
        blend_mode="overlay",
        p=1.0
    )
    
    # Apply transformation
    print("Applying transformation...")
    augmented = transform(image=image)["image"]
    
    # Create side-by-side comparison
    comparison = np.hstack([image, augmented])
    
    # Add labels
    cv2.putText(comparison, "BEFORE", (270, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(comparison, "AFTER", (910, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    # Add dividing line
    cv2.line(comparison, (640, 0), (640, 480), (0, 0, 0), 2)
    
    # Save result
    cv2.imwrite("simple_test_result.jpg", comparison)
    print("\n✅ Result saved as 'simple_test_result.jpg'")
    
    # Print summary
    print("\nTransformation applied:")
    print("- 2 red squares")
    print("- 2 green circles") 
    print("- 1 blue triangle")
    print("- 1 yellow rectangle")
    print(f"- Scale range: {transform.scale_range}")
    print(f"- Blend mode: {transform.blend_mode}")


if __name__ == "__main__":
    main()