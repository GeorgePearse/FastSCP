"""Create a test dataset with clearly colored shapes for visual testing."""

import json
import os
from pathlib import Path
from datetime import datetime
import cv2
import numpy as np


def create_colored_shapes_dataset(output_dir: str = "tests/colored_shapes"):
    """Create a dataset with distinct colored shapes on white backgrounds."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    images_dir = output_path / "images"
    images_dir.mkdir(exist_ok=True)
    
    # Define colors (BGR format for OpenCV)
    colors = {
        "red": (0, 0, 255),
        "green": (0, 255, 0),
        "blue": (255, 0, 0),
        "yellow": (0, 255, 255),
        "purple": (255, 0, 255),
        "cyan": (255, 255, 0),
    }
    
    # COCO dataset structure
    coco_dataset = {
        "info": {
            "description": "Colored Shapes Test Dataset",
            "version": "1.0",
            "year": 2025,
            "contributor": "FastSCP",
            "date_created": datetime.now().isoformat()
        },
        "licenses": [{"id": 1, "name": "Test License", "url": ""}],
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 1, "name": "red_square", "supercategory": "shape"},
            {"id": 2, "name": "green_circle", "supercategory": "shape"},
            {"id": 3, "name": "blue_triangle", "supercategory": "shape"},
            {"id": 4, "name": "yellow_rectangle", "supercategory": "shape"},
            {"id": 5, "name": "purple_star", "supercategory": "shape"},
            {"id": 6, "name": "cyan_hexagon", "supercategory": "shape"},
        ]
    }
    
    annotation_id = 1
    
    # Create individual shape images (for copy-paste sources)
    print("Creating individual shape images...")
    
    # 1. Red Square
    img = np.ones((200, 200, 3), dtype=np.uint8) * 255
    cv2.rectangle(img, (50, 50), (150, 150), colors["red"], -1)
    cv2.imwrite(str(images_dir / "shape_red_square.jpg"), img)
    
    coco_dataset["images"].append({
        "id": 1, "width": 200, "height": 200,
        "file_name": "shape_red_square.jpg"
    })
    coco_dataset["annotations"].append({
        "id": annotation_id, "image_id": 1, "category_id": 1,
        "bbox": [50, 50, 100, 100], "area": 10000,
        "segmentation": [[50, 50, 150, 50, 150, 150, 50, 150]],
        "iscrowd": 0
    })
    annotation_id += 1
    
    # 2. Green Circle
    img = np.ones((200, 200, 3), dtype=np.uint8) * 255
    cv2.circle(img, (100, 100), 50, colors["green"], -1)
    cv2.imwrite(str(images_dir / "shape_green_circle.jpg"), img)
    
    coco_dataset["images"].append({
        "id": 2, "width": 200, "height": 200,
        "file_name": "shape_green_circle.jpg"
    })
    coco_dataset["annotations"].append({
        "id": annotation_id, "image_id": 2, "category_id": 2,
        "bbox": [50, 50, 100, 100], "area": 10000,
        "segmentation": [[50, 50, 150, 50, 150, 150, 50, 150]],
        "iscrowd": 0
    })
    annotation_id += 1
    
    # 3. Blue Triangle
    img = np.ones((200, 200, 3), dtype=np.uint8) * 255
    triangle_pts = np.array([[100, 50], [50, 150], [150, 150]], np.int32)
    cv2.fillPoly(img, [triangle_pts], colors["blue"])
    cv2.imwrite(str(images_dir / "shape_blue_triangle.jpg"), img)
    
    coco_dataset["images"].append({
        "id": 3, "width": 200, "height": 200,
        "file_name": "shape_blue_triangle.jpg"
    })
    coco_dataset["annotations"].append({
        "id": annotation_id, "image_id": 3, "category_id": 3,
        "bbox": [50, 50, 100, 100], "area": 10000,
        "segmentation": [[100, 50, 50, 150, 150, 150]],
        "iscrowd": 0
    })
    annotation_id += 1
    
    # 4. Yellow Rectangle
    img = np.ones((200, 200, 3), dtype=np.uint8) * 255
    cv2.rectangle(img, (40, 70), (160, 130), colors["yellow"], -1)
    cv2.imwrite(str(images_dir / "shape_yellow_rectangle.jpg"), img)
    
    coco_dataset["images"].append({
        "id": 4, "width": 200, "height": 200,
        "file_name": "shape_yellow_rectangle.jpg"
    })
    coco_dataset["annotations"].append({
        "id": annotation_id, "image_id": 4, "category_id": 4,
        "bbox": [40, 70, 120, 60], "area": 7200,
        "segmentation": [[40, 70, 160, 70, 160, 130, 40, 130]],
        "iscrowd": 0
    })
    annotation_id += 1
    
    # 5. Purple Star (simplified as diamond)
    img = np.ones((200, 200, 3), dtype=np.uint8) * 255
    star_pts = np.array([[100, 40], [130, 100], [100, 160], [70, 100]], np.int32)
    cv2.fillPoly(img, [star_pts], colors["purple"])
    cv2.imwrite(str(images_dir / "shape_purple_star.jpg"), img)
    
    coco_dataset["images"].append({
        "id": 5, "width": 200, "height": 200,
        "file_name": "shape_purple_star.jpg"
    })
    coco_dataset["annotations"].append({
        "id": annotation_id, "image_id": 5, "category_id": 5,
        "bbox": [70, 40, 60, 120], "area": 7200,
        "segmentation": [[100, 40, 130, 100, 100, 160, 70, 100]],
        "iscrowd": 0
    })
    annotation_id += 1
    
    # 6. Cyan Hexagon
    img = np.ones((200, 200, 3), dtype=np.uint8) * 255
    hex_pts = np.array([
        [100, 40], [140, 65], [140, 115], 
        [100, 140], [60, 115], [60, 65]
    ], np.int32)
    cv2.fillPoly(img, [hex_pts], colors["cyan"])
    cv2.imwrite(str(images_dir / "shape_cyan_hexagon.jpg"), img)
    
    coco_dataset["images"].append({
        "id": 6, "width": 200, "height": 200,
        "file_name": "shape_cyan_hexagon.jpg"
    })
    coco_dataset["annotations"].append({
        "id": annotation_id, "image_id": 6, "category_id": 6,
        "bbox": [60, 40, 80, 100], "area": 8000,
        "segmentation": [[100, 40, 140, 65, 140, 115, 100, 140, 60, 115, 60, 65]],
        "iscrowd": 0
    })
    annotation_id += 1
    
    # Create test backgrounds with multiple shapes
    print("Creating test background images...")
    
    # Background 1: Simple gray
    bg1 = np.ones((640, 480, 3), dtype=np.uint8) * 200
    cv2.imwrite(str(images_dir / "background_gray.jpg"), bg1)
    coco_dataset["images"].append({
        "id": 7, "width": 640, "height": 480,
        "file_name": "background_gray.jpg"
    })
    
    # Background 2: Gradient
    bg2 = np.zeros((640, 480, 3), dtype=np.uint8)
    for i in range(480):
        bg2[i, :] = int(255 * (i / 480))
    cv2.imwrite(str(images_dir / "background_gradient.jpg"), bg2)
    coco_dataset["images"].append({
        "id": 8, "width": 640, "height": 480,
        "file_name": "background_gradient.jpg"
    })
    
    # Background 3: Pattern
    bg3 = np.ones((640, 480, 3), dtype=np.uint8) * 255
    for i in range(0, 640, 40):
        cv2.line(bg3, (i, 0), (i, 480), (220, 220, 220), 2)
    for i in range(0, 480, 40):
        cv2.line(bg3, (0, i), (640, i), (220, 220, 220), 2)
    cv2.imwrite(str(images_dir / "background_grid.jpg"), bg3)
    coco_dataset["images"].append({
        "id": 9, "width": 640, "height": 480,
        "file_name": "background_grid.jpg"
    })
    
    # Save COCO annotations
    with open(output_path / "annotations.json", 'w') as f:
        json.dump(coco_dataset, f, indent=2)
    
    print(f"\nDataset created successfully!")
    print(f"- {len(coco_dataset['images'])} images")
    print(f"- {len(coco_dataset['annotations'])} annotations")
    print(f"- {len(coco_dataset['categories'])} categories")
    print(f"Location: {output_path}")


if __name__ == "__main__":
    create_colored_shapes_dataset()