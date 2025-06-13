"""Generate synthetic test data for FastSCP testing."""

import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Dict, Any

import cv2
import numpy as np


@dataclass
class COCOImage:
    id: int
    width: int
    height: int
    file_name: str
    license: int = 1
    date_captured: str = ""


@dataclass
class COCOAnnotation:
    id: int
    image_id: int
    category_id: int
    bbox: List[float]  # [x, y, width, height]
    area: float
    segmentation: List[List[float]]
    iscrowd: int = 0


@dataclass
class COCOCategory:
    id: int
    name: str
    supercategory: str = "object"


class TestDataGenerator:
    """Generate synthetic images and COCO annotations for testing."""
    
    def __init__(self, output_dir: str = "tests/data", seed: int = 42):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir = self.output_dir / "images"
        self.images_dir.mkdir(exist_ok=True)
        
        np.random.seed(seed)
        self.annotation_id = 1
        
        # Define categories
        self.categories = [
            COCOCategory(id=1, name="rectangle", supercategory="shape"),
            COCOCategory(id=2, name="circle", supercategory="shape"),
            COCOCategory(id=3, name="triangle", supercategory="shape"),
        ]
    
    def draw_rectangle(self, image: np.ndarray, bbox: Tuple[int, int, int, int], 
                      color: Tuple[int, int, int]) -> np.ndarray:
        """Draw a filled rectangle on the image."""
        x, y, w, h = bbox
        cv2.rectangle(image, (x, y), (x + w, y + h), color, -1)
        return image
    
    def draw_circle(self, image: np.ndarray, center: Tuple[int, int], 
                   radius: int, color: Tuple[int, int, int]) -> np.ndarray:
        """Draw a filled circle on the image."""
        cv2.circle(image, center, radius, color, -1)
        return image
    
    def draw_triangle(self, image: np.ndarray, points: np.ndarray, 
                     color: Tuple[int, int, int]) -> np.ndarray:
        """Draw a filled triangle on the image."""
        cv2.fillPoly(image, [points], color)
        return image
    
    def generate_random_color(self) -> Tuple[int, int, int]:
        """Generate a random RGB color."""
        return tuple(int(x) for x in np.random.randint(50, 255, 3))
    
    def create_shape_annotation(self, shape_type: str, bbox: List[int], 
                               image_id: int) -> COCOAnnotation:
        """Create a COCO annotation for a shape."""
        x, y, w, h = bbox
        
        # Simple segmentation (rectangle outline)
        segmentation = [[x, y, x + w, y, x + w, y + h, x, y + h]]
        
        # Map shape type to category ID
        category_map = {"rectangle": 1, "circle": 2, "triangle": 3}
        category_id = category_map.get(shape_type, 1)
        
        annotation = COCOAnnotation(
            id=self.annotation_id,
            image_id=image_id,
            category_id=category_id,
            bbox=[float(x) for x in bbox],
            area=float(w * h),
            segmentation=segmentation
        )
        self.annotation_id += 1
        return annotation
    
    def generate_image_with_shapes(self, image_id: int, width: int = 640, 
                                  height: int = 480, num_shapes: int = 5) -> Tuple[np.ndarray, List[COCOAnnotation]]:
        """Generate a single image with random shapes."""
        # Create blank image with random background color
        bg_color = [int(x) for x in np.random.randint(200, 255, 3)]
        image = np.full((height, width, 3), bg_color, dtype=np.uint8)
        
        annotations = []
        
        for _ in range(num_shapes):
            shape_type = np.random.choice(["rectangle", "circle", "triangle"])
            color = self.generate_random_color()
            
            if shape_type == "rectangle":
                # Random rectangle parameters
                w = np.random.randint(30, min(150, width // 3))
                h = np.random.randint(30, min(150, height // 3))
                x = np.random.randint(0, width - w)
                y = np.random.randint(0, height - h)
                
                self.draw_rectangle(image, (x, y, w, h), color)
                annotations.append(self.create_shape_annotation("rectangle", [x, y, w, h], image_id))
                
            elif shape_type == "circle":
                # Random circle parameters
                radius = np.random.randint(20, min(80, width // 4, height // 4))
                cx = np.random.randint(radius, width - radius)
                cy = np.random.randint(radius, height - radius)
                
                self.draw_circle(image, (cx, cy), radius, color)
                # Approximate bounding box for circle
                bbox = [cx - radius, cy - radius, 2 * radius, 2 * radius]
                annotations.append(self.create_shape_annotation("circle", bbox, image_id))
                
            else:  # triangle
                # Random triangle parameters
                size = np.random.randint(30, min(100, width // 3, height // 3))
                cx = np.random.randint(size, width - size)
                cy = np.random.randint(size, height - size)
                
                # Create equilateral triangle points
                points = np.array([
                    [cx, cy - size],
                    [cx - int(size * 0.866), cy + size // 2],
                    [cx + int(size * 0.866), cy + size // 2]
                ], dtype=np.int32)
                
                self.draw_triangle(image, points, color)
                # Bounding box for triangle
                x_min, y_min = points.min(axis=0)
                x_max, y_max = points.max(axis=0)
                bbox = [int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)]
                annotations.append(self.create_shape_annotation("triangle", bbox, image_id))
        
        return image, annotations
    
    def generate_dataset(self, num_images: int = 5, shapes_per_image: int = 4) -> Dict[str, Any]:
        """Generate a complete test dataset with COCO annotations."""
        images = []
        annotations = []
        
        print(f"Generating {num_images} test images...")
        
        for i in range(num_images):
            # Vary image dimensions slightly
            width = 640 + (i % 3) * 64
            height = 480 + (i % 2) * 64
            
            # Generate image
            image_array, image_annotations = self.generate_image_with_shapes(
                image_id=i + 1,
                width=width,
                height=height,
                num_shapes=shapes_per_image + (i % 3)  # Vary number of shapes
            )
            
            # Save image
            filename = f"test_image_{i+1:03d}.jpg"
            filepath = self.images_dir / filename
            cv2.imwrite(str(filepath), image_array)
            
            # Add image metadata
            images.append(COCOImage(
                id=i + 1,
                width=width,
                height=height,
                file_name=filename,
                date_captured=datetime.now().isoformat()
            ))
            
            # Add annotations
            annotations.extend(image_annotations)
        
        # Create COCO format dataset
        coco_dataset = {
            "info": {
                "description": "FastSCP Test Dataset",
                "version": "1.0",
                "year": 2025,
                "contributor": "FastSCP",
                "date_created": datetime.now().isoformat()
            },
            "licenses": [
                {
                    "id": 1,
                    "name": "Test License",
                    "url": ""
                }
            ],
            "images": [asdict(img) for img in images],
            "annotations": [asdict(ann) for ann in annotations],
            "categories": [asdict(cat) for cat in self.categories]
        }
        
        # Save annotations
        annotations_path = self.output_dir / "annotations.json"
        with open(annotations_path, 'w') as f:
            json.dump(coco_dataset, f, indent=2)
        
        print(f"Generated dataset with {len(images)} images and {len(annotations)} annotations")
        print(f"Images saved to: {self.images_dir}")
        print(f"Annotations saved to: {annotations_path}")
        
        return coco_dataset


def main():
    """Generate test data when run as script."""
    generator = TestDataGenerator()
    generator.generate_dataset(num_images=5, shapes_per_image=4)


if __name__ == "__main__":
    main()