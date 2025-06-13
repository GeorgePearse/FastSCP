"""Tests for data generator module."""

import json
import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from tests.data_generator import TestDataGenerator, COCOImage, COCOAnnotation, COCOCategory


class TestDataGeneratorUnit:
    """Unit tests for TestDataGenerator."""
    
    def test_init(self, tmp_path):
        """Test generator initialization."""
        generator = TestDataGenerator(output_dir=str(tmp_path))
        assert generator.output_dir.exists()
        assert generator.images_dir.exists()
        assert generator.annotation_id == 1
        assert len(generator.categories) == 3
    
    def test_draw_rectangle(self):
        """Test rectangle drawing."""
        generator = TestDataGenerator()
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        color = (255, 0, 0)
        bbox = (10, 10, 30, 40)
        
        result = generator.draw_rectangle(image, bbox, color)
        
        # Check that rectangle was drawn
        assert np.all(result[10:50, 10:40] == color)
        assert np.all(result[0:10, :] == 0)  # Outside rectangle
    
    def test_draw_circle(self):
        """Test circle drawing."""
        generator = TestDataGenerator()
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        color = (0, 255, 0)
        center = (50, 50)
        radius = 20
        
        result = generator.draw_circle(image, center, radius, color)
        
        # Check center pixel
        assert np.all(result[50, 50] == color)
        # Check outside radius
        assert np.all(result[0, 0] == 0)
    
    def test_draw_triangle(self):
        """Test triangle drawing."""
        generator = TestDataGenerator()
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        color = (0, 0, 255)
        points = np.array([[50, 20], [30, 60], [70, 60]])
        
        result = generator.draw_triangle(image, points, color)
        
        # Check that some pixels inside triangle are colored
        assert np.any(result[40, 50] == color)
        # Check outside triangle
        assert np.all(result[0, 0] == 0)
    
    def test_generate_random_color(self):
        """Test random color generation."""
        generator = TestDataGenerator()
        
        colors = [generator.generate_random_color() for _ in range(10)]
        
        for color in colors:
            assert isinstance(color, tuple)
            assert len(color) == 3
            assert all(50 <= c <= 255 for c in color)
    
    def test_create_shape_annotation(self):
        """Test COCO annotation creation."""
        generator = TestDataGenerator()
        bbox = [10, 20, 30, 40]
        
        ann = generator.create_shape_annotation("rectangle", bbox, image_id=1)
        
        assert ann.id == 1
        assert ann.image_id == 1
        assert ann.category_id == 1  # rectangle
        assert ann.bbox == [10.0, 20.0, 30.0, 40.0]
        assert ann.area == 1200.0
        assert len(ann.segmentation) == 1
        assert len(ann.segmentation[0]) == 8  # 4 points * 2 coords
    
    def test_generate_image_with_shapes(self):
        """Test single image generation."""
        generator = TestDataGenerator()
        
        image, annotations = generator.generate_image_with_shapes(
            image_id=1, width=320, height=240, num_shapes=3
        )
        
        assert image.shape == (240, 320, 3)
        assert len(annotations) == 3
        assert all(ann.image_id == 1 for ann in annotations)
        
        # Check that image is not all one color
        assert len(np.unique(image.reshape(-1, 3), axis=0)) > 1


class TestDataGeneratorIntegration:
    """Integration tests for full dataset generation."""
    
    def test_generate_dataset(self, tmp_path):
        """Test complete dataset generation."""
        generator = TestDataGenerator(output_dir=str(tmp_path), seed=42)
        
        dataset = generator.generate_dataset(num_images=3, shapes_per_image=2)
        
        # Check dataset structure
        assert "info" in dataset
        assert "licenses" in dataset
        assert "images" in dataset
        assert "annotations" in dataset
        assert "categories" in dataset
        
        # Check counts
        assert len(dataset["images"]) == 3
        assert len(dataset["annotations"]) >= 6  # At least 2 per image
        assert len(dataset["categories"]) == 3
        
        # Check files exist
        assert (tmp_path / "annotations.json").exists()
        assert len(list((tmp_path / "images").glob("*.jpg"))) == 3
        
        # Validate JSON structure
        with open(tmp_path / "annotations.json") as f:
            loaded = json.load(f)
            assert loaded["info"]["description"] == "FastSCP Test Dataset"
    
    def test_generated_images_are_valid(self, tmp_path):
        """Test that generated images can be loaded."""
        generator = TestDataGenerator(output_dir=str(tmp_path))
        generator.generate_dataset(num_images=2)
        
        # Try to load each image
        for img_file in (tmp_path / "images").glob("*.jpg"):
            img = cv2.imread(str(img_file))
            assert img is not None
            assert img.shape[2] == 3  # RGB channels
            assert img.dtype == np.uint8