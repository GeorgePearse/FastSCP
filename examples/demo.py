"""Demo script showing FastSCP usage."""

import time
from pathlib import Path

import cv2
import numpy as np
import albumentations as A

# Ensure test data exists
import sys

sys.path.append(str(Path(__file__).parent.parent))

from fastscp import SimpleCopyPaste
from tests.data_generator import TestDataGenerator


def ensure_test_data():
    """Ensure test data exists for demo."""
    data_dir = Path(__file__).parent.parent / "tests" / "data"
    if not (data_dir / "annotations.json").exists():
        print("Generating test data...")
        generator = TestDataGenerator()
        generator.generate_dataset(num_images=5, shapes_per_image=4)
    return data_dir


def main():
    """Run demo of SimpleCopyPaste transform."""
    print("FastSCP Demo\n" + "=" * 50)

    # Ensure test data exists
    data_dir = ensure_test_data()

    # Create transform
    print("\n1. Creating SimpleCopyPaste transform...")
    transform = SimpleCopyPaste(
        coco_file=str(data_dir / "annotations.json"),
        object_counts={"rectangle": 3, "circle": 2, "triangle": 1},
        blend_mode="overlay",
        scale_range=(0.5, 1.5),
        p=1.0,
    )
    print("   ✓ Transform created")

    # Create or load test image
    print("\n2. Creating test image...")
    image = np.ones((640, 480, 3), dtype=np.uint8) * 200  # Light gray background

    # Add some variation to make it more interesting
    cv2.rectangle(image, (50, 50), (200, 150), (100, 150, 200), -1)
    cv2.circle(image, (400, 300), 80, (200, 150, 100), -1)

    print("   ✓ Test image created (640x480)")

    # Apply augmentation
    print("\n3. Applying copy-paste augmentation...")
    start_time = time.time()
    augmented = transform.apply(image)
    elapsed = (time.time() - start_time) * 1000
    print(f"   ✓ Augmentation completed in {elapsed:.2f} ms")

    # Save results
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)

    cv2.imwrite(str(output_dir / "original.jpg"), image)
    cv2.imwrite(str(output_dir / "augmented.jpg"), augmented)
    print(f"\n4. Results saved to {output_dir}")

    # Demo with Albumentations pipeline
    print("\n5. Demo with full Albumentations pipeline...")
    pipeline = A.Compose(
        [
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
            SimpleCopyPaste(
                coco_file=str(data_dir / "annotations.json"),
                object_counts={"rectangle": 2, "circle": 2},
                blend_mode="mix",
                p=0.8,
            ),
            A.HorizontalFlip(p=0.5),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        ]
    )

    # Apply pipeline multiple times
    for i in range(3):
        result = pipeline(image=image)
        augmented = result["image"]
        cv2.imwrite(str(output_dir / f"pipeline_{i+1}.jpg"), augmented)

    print("   ✓ Pipeline results saved")

    # Performance test
    print("\n6. Performance benchmark...")
    times = []
    for _ in range(50):
        start = time.time()
        transform.apply(image)
        times.append(time.time() - start)

    avg_time = np.mean(times) * 1000
    std_time = np.std(times) * 1000
    print(f"   Average time: {avg_time:.2f} ± {std_time:.2f} ms")
    print(f"   Min/Max: {np.min(times)*1000:.2f} / {np.max(times)*1000:.2f} ms")

    print("\n" + "=" * 50)
    print("Demo completed! Check the output directory for results.")


if __name__ == "__main__":
    main()
