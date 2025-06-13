"""Tests for COCO loader module."""

import numpy as np
import pytest

from fastscp.coco_loader_segmented import COCOLoaderSegmented
from tests.data_generator import TestDataGenerator


@pytest.fixture
def test_dataset(tmp_path):
    """Generate test dataset for loader tests."""
    generator = TestDataGenerator(output_dir=str(tmp_path), seed=42)
    dataset = generator.generate_dataset(num_images=3, shapes_per_image=3)
    return {
        "annotation_file": str(tmp_path / "annotations.json"),
        "image_dir": str(tmp_path / "images"),
        "dataset": dataset,
    }


class TestCOCOLoaderInit:
    """Test COCOLoaderSegmented initialization."""

    def test_init_success(self, test_dataset):
        """Test successful initialization."""
        loader = COCOLoaderSegmented(test_dataset["annotation_file"])

        assert loader.annotation_file.exists()
        assert loader.image_dir.exists()
        assert loader.cache_size == 1000
        assert len(loader._cache) == 0
        assert len(loader.categories) == 3

    def test_init_with_custom_image_dir(self, test_dataset):
        """Test initialization with custom image directory."""
        loader = COCOLoaderSegmented(
            test_dataset["annotation_file"], image_dir=test_dataset["image_dir"]
        )

        assert str(loader.image_dir) == test_dataset["image_dir"]

    def test_init_missing_annotation_file(self):
        """Test initialization with missing annotation file."""
        with pytest.raises(FileNotFoundError, match="Annotation file not found"):
            COCOLoaderSegmented("nonexistent.json")

    def test_init_missing_image_dir(self, tmp_path):
        """Test initialization with missing image directory."""
        # Create annotation file but no images directory
        ann_file = tmp_path / "annotations.json"
        ann_file.write_text("{}")

        with pytest.raises(FileNotFoundError, match="Image directory not found"):
            COCOLoaderSegmented(str(ann_file))


class TestCOCOLoaderMethods:
    """Test COCOLoaderSegmented methods."""

    def test_get_category_id(self, test_dataset):
        """Test category ID lookup."""
        loader = COCOLoaderSegmented(test_dataset["annotation_file"])

        assert loader.get_category_id("rectangle") == 1
        assert loader.get_category_id("circle") == 2
        assert loader.get_category_id("triangle") == 3
        assert loader.get_category_id("nonexistent") is None

    def test_load_object_crop(self, test_dataset):
        """Test loading object crop."""
        loader = COCOLoaderSegmented(test_dataset["annotation_file"])

        # Get first annotation ID
        ann_id = list(loader.coco.anns.keys())[0]

        # Load crop
        crop = loader.load_object_crop(ann_id)

        assert crop is not None
        assert isinstance(crop, np.ndarray)
        assert crop.ndim == 3
        assert crop.shape[2] == 3

        # Check cache
        assert ann_id in loader._cache
        assert len(loader._cache) == 1

    def test_load_object_crop_invalid_id(self, test_dataset):
        """Test loading with invalid annotation ID."""
        loader = COCOLoaderSegmented(test_dataset["annotation_file"])

        crop = loader.load_object_crop(99999)
        assert crop is None

    def test_get_random_objects(self, test_dataset):
        """Test getting random objects by category."""
        loader = COCOLoaderSegmented(test_dataset["annotation_file"])

        # Get rectangles
        rectangles = loader.get_random_objects("rectangle", count=2)

        assert len(rectangles) <= 2
        assert all(isinstance(obj, np.ndarray) for obj in rectangles)

        # Get from non-existent category
        empty = loader.get_random_objects("nonexistent", count=1)
        assert len(empty) == 0

    def test_get_all_categories(self, test_dataset):
        """Test getting all categories."""
        loader = COCOLoaderSegmented(test_dataset["annotation_file"])

        categories = loader.get_all_categories()

        assert len(categories) == 3
        assert categories[1] == "rectangle"
        assert categories[2] == "circle"
        assert categories[3] == "triangle"

    def test_get_annotation_count(self, test_dataset):
        """Test annotation counting."""
        loader = COCOLoaderSegmented(test_dataset["annotation_file"])
        dataset = test_dataset["dataset"]

        # Total count
        total = loader.get_annotation_count()
        assert total == len(dataset["annotations"])

        # Category count
        rect_count = loader.get_annotation_count("rectangle")
        assert rect_count >= 0
        assert rect_count <= total

        # Non-existent category
        assert loader.get_annotation_count("nonexistent") == 0


class TestCOCOLoaderCache:
    """Test caching functionality."""

    def test_cache_reuse(self, test_dataset):
        """Test that cached objects are reused."""
        loader = COCOLoaderSegmented(test_dataset["annotation_file"], cache_size=10)

        ann_id = list(loader.coco.anns.keys())[0]

        # First load
        crop1 = loader.load_object_crop(ann_id)
        cache_size_1 = len(loader._cache)

        # Second load (should use cache)
        crop2 = loader.load_object_crop(ann_id)
        cache_size_2 = len(loader._cache)

        assert cache_size_1 == cache_size_2 == 1
        assert np.array_equal(crop1, crop2)

    def test_cache_eviction(self, test_dataset):
        """Test LRU cache eviction."""
        loader = COCOLoaderSegmented(test_dataset["annotation_file"], cache_size=2)

        ann_ids = list(loader.coco.anns.keys())[:3]

        # Load 3 objects with cache size 2
        for ann_id in ann_ids:
            loader.load_object_crop(ann_id)

        # Cache should have only last 2
        assert len(loader._cache) == 2
        assert ann_ids[0] not in loader._cache
        assert ann_ids[1] in loader._cache
        assert ann_ids[2] in loader._cache

    def test_clear_cache(self, test_dataset):
        """Test cache clearing."""
        loader = COCOLoaderSegmented(test_dataset["annotation_file"])

        # Load some objects
        ann_id = list(loader.coco.anns.keys())[0]
        loader.load_object_crop(ann_id)

        assert len(loader._cache) > 0

        # Clear cache
        loader.clear_cache()

        assert len(loader._cache) == 0
        assert len(loader._cache_order) == 0

    def test_get_cache_info(self, test_dataset):
        """Test cache info retrieval."""
        loader = COCOLoaderSegmented(test_dataset["annotation_file"], cache_size=100)

        info = loader.get_cache_info()

        assert info["size"] == 0
        assert info["max_size"] == 100
        assert "hit_rate" in info
