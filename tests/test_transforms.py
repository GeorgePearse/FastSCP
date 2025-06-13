"""Tests for transforms module."""

import time
from pathlib import Path

import cv2
import numpy as np
import pytest
import albumentations as A

from fastscp.transforms import SimpleCopyPaste
from tests.data_generator import TestDataGenerator


@pytest.fixture
def test_dataset(tmp_path):
    """Generate test dataset for transform tests."""
    generator = TestDataGenerator(output_dir=str(tmp_path), seed=42)
    dataset = generator.generate_dataset(num_images=5, shapes_per_image=4)
    return {
        "annotation_file": str(tmp_path / "annotations.json"),
        "image_dir": str(tmp_path / "images"),
        "dataset": dataset
    }


class TestSimpleCopyPasteInit:
    """Test SimpleCopyPaste initialization."""
    
    def test_init_success(self, test_dataset):
        """Test successful initialization."""
        transform = SimpleCopyPaste(
            coco_file=test_dataset["annotation_file"],
            object_counts={"rectangle": 2, "circle": 1},
            p=1.0
        )
        
        assert transform.coco_loader is not None
        assert transform.object_counts == {"rectangle": 2, "circle": 1}
        assert transform.blend_mode == "overlay"
        assert transform.scale_range == (0.5, 1.5)
        assert transform.min_visibility == 0.3
        assert transform.p == 1.0
    
    def test_init_with_custom_params(self, test_dataset):
        """Test initialization with custom parameters."""
        transform = SimpleCopyPaste(
            coco_file=test_dataset["annotation_file"],
            object_counts={"triangle": 3},
            blend_mode="mix",
            scale_range=(0.8, 1.2),
            min_visibility=0.5,
            p=0.7
        )
        
        assert transform.blend_mode == "mix"
        assert transform.scale_range == (0.8, 1.2)
        assert transform.min_visibility == 0.5
        assert transform.p == 0.7
    
    def test_init_invalid_category(self, test_dataset):
        """Test initialization with invalid category."""
        with pytest.raises(ValueError, match="Category 'invalid' not found"):
            SimpleCopyPaste(
                coco_file=test_dataset["annotation_file"],
                object_counts={"invalid": 1}
            )


class TestSimpleCopyPasteApply:
    """Test SimpleCopyPaste apply method."""
    
    def test_apply_basic(self, test_dataset):
        """Test basic apply functionality."""
        transform = SimpleCopyPaste(
            coco_file=test_dataset["annotation_file"],
            object_counts={"rectangle": 2},
            p=1.0
        )
        
        # Create test image
        image = np.ones((256, 256, 3), dtype=np.uint8) * 128
        
        # Apply transform
        result = transform.apply(image)
        
        assert result.shape == image.shape
        assert result.dtype == image.dtype
        # Check that image was modified (not all pixels are 128)
        assert not np.all(result == 128)
    
    def test_apply_no_objects(self, test_dataset):
        """Test apply with zero objects."""
        transform = SimpleCopyPaste(
            coco_file=test_dataset["annotation_file"],
            object_counts={"rectangle": 0},
            p=1.0
        )
        
        image = np.ones((256, 256, 3), dtype=np.uint8) * 128
        result = transform.apply(image)
        
        # Image should be unchanged
        assert np.array_equal(result, image)
    
    def test_apply_multiple_categories(self, test_dataset):
        """Test apply with multiple categories."""
        transform = SimpleCopyPaste(
            coco_file=test_dataset["annotation_file"],
            object_counts={"rectangle": 1, "circle": 1, "triangle": 1},
            p=1.0
        )
        
        image = np.ones((256, 256, 3), dtype=np.uint8) * 128
        result = transform.apply(image)
        
        assert result.shape == image.shape
        assert not np.array_equal(result, image)
    
    def test_blend_modes(self, test_dataset):
        """Test different blend modes."""
        image = np.ones((256, 256, 3), dtype=np.uint8) * 128
        
        # Overlay mode
        transform_overlay = SimpleCopyPaste(
            coco_file=test_dataset["annotation_file"],
            object_counts={"rectangle": 1},
            blend_mode="overlay",
            p=1.0
        )
        result_overlay = transform_overlay.apply(image)
        
        # Mix mode
        transform_mix = SimpleCopyPaste(
            coco_file=test_dataset["annotation_file"],
            object_counts={"rectangle": 1},
            blend_mode="mix",
            p=1.0
        )
        result_mix = transform_mix.apply(image)
        
        # Both should modify the image
        assert not np.array_equal(result_overlay, image)
        assert not np.array_equal(result_mix, image)


class TestSimpleCopyPasteIntegration:
    """Integration tests with Albumentations."""
    
    def test_albumentations_pipeline(self, test_dataset):
        """Test integration with Albumentations pipeline."""
        transform = A.Compose([
            SimpleCopyPaste(
                coco_file=test_dataset["annotation_file"],
                object_counts={"rectangle": 2, "circle": 1},
                p=1.0
            ),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
        ])
        
        image = np.ones((256, 256, 3), dtype=np.uint8) * 128
        
        # Apply multiple times to test randomness
        results = []
        for _ in range(5):
            augmented = transform(image=image)
            results.append(augmented["image"])
        
        # Check that we get different results (due to randomness)
        assert not all(np.array_equal(results[0], r) for r in results[1:])
    
    def test_with_real_image(self, test_dataset):
        """Test with a real test image."""
        # Load one of the generated test images
        test_image_path = Path(test_dataset["image_dir"]) / "test_image_001.jpg"
        image = cv2.imread(str(test_image_path))
        
        transform = SimpleCopyPaste(
            coco_file=test_dataset["annotation_file"],
            object_counts={"rectangle": 3, "circle": 2},
            p=1.0
        )
        
        result = transform.apply(image)
        
        assert result.shape == image.shape
        assert result.dtype == image.dtype
        assert not np.array_equal(result, image)


class TestPerformance:
    """Performance benchmarks."""
    
    def test_single_image_performance(self, test_dataset):
        """Test performance on single image."""
        transform = SimpleCopyPaste(
            coco_file=test_dataset["annotation_file"],
            object_counts={"rectangle": 2, "circle": 1},
            p=1.0
        )
        
        image = np.ones((512, 512, 3), dtype=np.uint8) * 128
        
        # Warm up
        transform.apply(image)
        
        # Time multiple runs
        times = []
        for _ in range(10):
            start = time.time()
            transform.apply(image)
            times.append(time.time() - start)
        
        avg_time = np.mean(times)
        print(f"\nAverage time per 512x512 image: {avg_time*1000:.2f} ms")
        
        # Should be fast (< 50ms for basic implementation)
        assert avg_time < 0.05
    
    def test_batch_performance(self, test_dataset):
        """Test performance on batch of images."""
        transform = SimpleCopyPaste(
            coco_file=test_dataset["annotation_file"],
            object_counts={"rectangle": 2, "circle": 1},
            p=1.0
        )
        
        batch_size = 32
        images = [np.ones((256, 256, 3), dtype=np.uint8) * 128 for _ in range(batch_size)]
        
        start = time.time()
        for img in images:
            transform.apply(img)
        total_time = time.time() - start
        
        print(f"\nTime for batch of {batch_size} 256x256 images: {total_time*1000:.2f} ms")
        print(f"Average time per image: {total_time/batch_size*1000:.2f} ms")
        
        # Should process batch quickly
        assert total_time < 1.0  # Less than 1 second for 32 images