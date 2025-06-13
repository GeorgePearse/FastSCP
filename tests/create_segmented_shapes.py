"""Create a test dataset with precise segmentation polygons for each shape."""

import json
import os
from pathlib import Path
from datetime import datetime
import cv2
import numpy as np
from typing import List, Tuple


def generate_circle_polygon(center_x: int, center_y: int, radius: int, num_points: int = 32) -> List[float]:
    """Generate polygon points for a circle."""
    points = []
    for i in range(num_points):
        angle = 2 * np.pi * i / num_points
        x = center_x + radius * np.cos(angle)
        y = center_y + radius * np.sin(angle)
        points.extend([x, y])
    return points


def generate_regular_polygon(center_x: int, center_y: int, radius: int, num_sides: int, rotation: float = 0) -> List[float]:
    """Generate polygon points for a regular polygon."""
    points = []
    for i in range(num_sides):
        angle = 2 * np.pi * i / num_sides + rotation
        x = center_x + radius * np.cos(angle)
        y = center_y + radius * np.sin(angle)
        points.extend([x, y])
    return points


def create_segmented_shapes_dataset(output_dir: str = "tests/segmented_shapes"):
    """Create a dataset with precise segmentation polygons."""
    
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
        "orange": (0, 165, 255),
        "pink": (203, 192, 255),
    }
    
    # COCO dataset structure
    coco_dataset = {
        "info": {
            "description": "Segmented Shapes Test Dataset",
            "version": "1.0",
            "year": 2025,
            "contributor": "FastSCP",
            "date_created": datetime.now().isoformat()
        },
        "licenses": [{"id": 1, "name": "Test License", "url": ""}],
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 1, "name": "square", "supercategory": "shape"},
            {"id": 2, "name": "circle", "supercategory": "shape"},
            {"id": 3, "name": "triangle", "supercategory": "shape"},
            {"id": 4, "name": "pentagon", "supercategory": "shape"},
            {"id": 5, "name": "hexagon", "supercategory": "shape"},
            {"id": 6, "name": "star", "supercategory": "shape"},
            {"id": 7, "name": "diamond", "supercategory": "shape"},
            {"id": 8, "name": "octagon", "supercategory": "shape"},
        ]
    }
    
    annotation_id = 1
    image_id = 1
    
    print("Creating segmented shape images...")
    
    # Create images with multiple shapes
    for img_idx in range(5):
        # Create blank image
        width, height = 800, 600
        img = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        # Add image to dataset
        filename = f"multi_shapes_{img_idx:03d}.jpg"
        coco_dataset["images"].append({
            "id": image_id,
            "width": width,
            "height": height,
            "file_name": filename
        })
        
        # Add various shapes
        shapes_config = [
            # (shape_type, center_x, center_y, size, color_name)
            ("square", 150, 150, 80, "red"),
            ("circle", 400, 150, 60, "green"),
            ("triangle", 650, 150, 70, "blue"),
            ("pentagon", 150, 400, 65, "yellow"),
            ("hexagon", 400, 400, 60, "purple"),
            ("star", 650, 400, 75, "cyan"),
        ]
        
        for shape_type, cx, cy, size, color_name in shapes_config:
            color = colors[color_name]
            
            if shape_type == "square":
                # Square with precise corners
                half_size = size // 2
                pts = np.array([
                    [cx - half_size, cy - half_size],
                    [cx + half_size, cy - half_size],
                    [cx + half_size, cy + half_size],
                    [cx - half_size, cy + half_size]
                ], np.int32)
                cv2.fillPoly(img, [pts], color)
                
                # Segmentation
                segmentation = [[float(p) for point in pts for p in point]]
                bbox = [cx - half_size, cy - half_size, size, size]
                category_id = 1
                
            elif shape_type == "circle":
                # Circle
                cv2.circle(img, (cx, cy), size, color, -1)
                
                # Segmentation (polygon approximation)
                segmentation = [generate_circle_polygon(cx, cy, size, 32)]
                bbox = [cx - size, cy - size, 2 * size, 2 * size]
                category_id = 2
                
            elif shape_type == "triangle":
                # Equilateral triangle
                pts = np.array([
                    [cx, cy - size],
                    [cx - int(size * 0.866), cy + size // 2],
                    [cx + int(size * 0.866), cy + size // 2]
                ], np.int32)
                cv2.fillPoly(img, [pts], color)
                
                # Segmentation
                segmentation = [[float(p) for point in pts for p in point]]
                x_min, y_min = pts.min(axis=0)
                x_max, y_max = pts.max(axis=0)
                bbox = [int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)]
                category_id = 3
                
            elif shape_type == "pentagon":
                # Pentagon
                pts_list = generate_regular_polygon(cx, cy, size, 5, -np.pi/2)
                pts = np.array([[pts_list[i], pts_list[i+1]] for i in range(0, len(pts_list), 2)], np.int32)
                cv2.fillPoly(img, [pts], color)
                
                # Segmentation
                segmentation = [pts_list]
                x_coords = pts[:, 0]
                y_coords = pts[:, 1]
                bbox = [int(x_coords.min()), int(y_coords.min()), 
                       int(x_coords.max() - x_coords.min()), int(y_coords.max() - y_coords.min())]
                category_id = 4
                
            elif shape_type == "hexagon":
                # Hexagon
                pts_list = generate_regular_polygon(cx, cy, size, 6, 0)
                pts = np.array([[pts_list[i], pts_list[i+1]] for i in range(0, len(pts_list), 2)], np.int32)
                cv2.fillPoly(img, [pts], color)
                
                # Segmentation
                segmentation = [pts_list]
                x_coords = pts[:, 0]
                y_coords = pts[:, 1]
                bbox = [int(x_coords.min()), int(y_coords.min()), 
                       int(x_coords.max() - x_coords.min()), int(y_coords.max() - y_coords.min())]
                category_id = 5
                
            elif shape_type == "star":
                # 5-pointed star
                outer_pts = []
                inner_pts = []
                for i in range(5):
                    # Outer points
                    angle = 2 * np.pi * i / 5 - np.pi / 2
                    outer_pts.append([cx + size * np.cos(angle), cy + size * np.sin(angle)])
                    # Inner points
                    angle = 2 * np.pi * (i + 0.5) / 5 - np.pi / 2
                    inner_pts.append([cx + size * 0.4 * np.cos(angle), cy + size * 0.4 * np.sin(angle)])
                
                # Interleave points
                star_pts = []
                for i in range(5):
                    star_pts.append(outer_pts[i])
                    star_pts.append(inner_pts[i])
                
                pts = np.array(star_pts, np.int32)
                cv2.fillPoly(img, [pts], color)
                
                # Segmentation
                segmentation = [[float(p) for point in star_pts for p in point]]
                x_coords = pts[:, 0]
                y_coords = pts[:, 1]
                bbox = [int(x_coords.min()), int(y_coords.min()), 
                       int(x_coords.max() - x_coords.min()), int(y_coords.max() - y_coords.min())]
                category_id = 6
            
            # Add annotation
            coco_dataset["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category_id,
                "bbox": [float(b) for b in bbox],
                "area": float(bbox[2] * bbox[3]),
                "segmentation": segmentation,
                "iscrowd": 0
            })
            annotation_id += 1
        
        # Save image
        cv2.imwrite(str(images_dir / filename), img)
        image_id += 1
    
    # Create individual shape images with transparent backgrounds
    print("Creating individual shape images with masks...")
    
    for shape_idx, (shape_name, cat_id) in enumerate([
        ("square", 1), ("circle", 2), ("triangle", 3), 
        ("pentagon", 4), ("hexagon", 5), ("star", 6),
        ("diamond", 7), ("octagon", 8)
    ]):
        # Create image and mask
        img = np.ones((300, 300, 3), dtype=np.uint8) * 255
        mask = np.zeros((300, 300), dtype=np.uint8)
        
        cx, cy = 150, 150
        size = 100
        color = list(colors.values())[shape_idx % len(colors)]
        
        if shape_name == "square":
            half_size = size // 2
            pts = np.array([
                [cx - half_size, cy - half_size],
                [cx + half_size, cy - half_size],
                [cx + half_size, cy + half_size],
                [cx - half_size, cy + half_size]
            ], np.int32)
            cv2.fillPoly(img, [pts], color)
            cv2.fillPoly(mask, [pts], 255)
            segmentation = [[float(p) for point in pts for p in point]]
            bbox = [cx - half_size, cy - half_size, size, size]
            
        elif shape_name == "circle":
            cv2.circle(img, (cx, cy), size, color, -1)
            cv2.circle(mask, (cx, cy), size, 255, -1)
            segmentation = [generate_circle_polygon(cx, cy, size, 32)]
            bbox = [cx - size, cy - size, 2 * size, 2 * size]
            
        elif shape_name == "triangle":
            pts = np.array([
                [cx, cy - size],
                [cx - int(size * 0.866), cy + size // 2],
                [cx + int(size * 0.866), cy + size // 2]
            ], np.int32)
            cv2.fillPoly(img, [pts], color)
            cv2.fillPoly(mask, [pts], 255)
            segmentation = [[float(p) for point in pts for p in point]]
            x_min, y_min = pts.min(axis=0)
            x_max, y_max = pts.max(axis=0)
            bbox = [int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)]
            
        elif shape_name == "pentagon":
            pts_list = generate_regular_polygon(cx, cy, size, 5, -np.pi/2)
            pts = np.array([[pts_list[i], pts_list[i+1]] for i in range(0, len(pts_list), 2)], np.int32)
            cv2.fillPoly(img, [pts], color)
            cv2.fillPoly(mask, [pts], 255)
            segmentation = [pts_list]
            x_coords = pts[:, 0]
            y_coords = pts[:, 1]
            bbox = [int(x_coords.min()), int(y_coords.min()), 
                   int(x_coords.max() - x_coords.min()), int(y_coords.max() - y_coords.min())]
            
        elif shape_name == "hexagon":
            pts_list = generate_regular_polygon(cx, cy, size, 6, 0)
            pts = np.array([[pts_list[i], pts_list[i+1]] for i in range(0, len(pts_list), 2)], np.int32)
            cv2.fillPoly(img, [pts], color)
            cv2.fillPoly(mask, [pts], 255)
            segmentation = [pts_list]
            x_coords = pts[:, 0]
            y_coords = pts[:, 1]
            bbox = [int(x_coords.min()), int(y_coords.min()), 
                   int(x_coords.max() - x_coords.min()), int(y_coords.max() - y_coords.min())]
            
        elif shape_name == "star":
            outer_pts = []
            inner_pts = []
            for i in range(5):
                angle = 2 * np.pi * i / 5 - np.pi / 2
                outer_pts.append([cx + size * np.cos(angle), cy + size * np.sin(angle)])
                angle = 2 * np.pi * (i + 0.5) / 5 - np.pi / 2
                inner_pts.append([cx + size * 0.4 * np.cos(angle), cy + size * 0.4 * np.sin(angle)])
            star_pts = []
            for i in range(5):
                star_pts.append(outer_pts[i])
                star_pts.append(inner_pts[i])
            pts = np.array(star_pts, np.int32)
            cv2.fillPoly(img, [pts], color)
            cv2.fillPoly(mask, [pts], 255)
            segmentation = [[float(p) for point in star_pts for p in point]]
            x_coords = pts[:, 0]
            y_coords = pts[:, 1]
            bbox = [int(x_coords.min()), int(y_coords.min()), 
                   int(x_coords.max() - x_coords.min()), int(y_coords.max() - y_coords.min())]
            
        elif shape_name == "diamond":
            pts = np.array([
                [cx, cy - size],
                [cx + size * 0.7, cy],
                [cx, cy + size],
                [cx - size * 0.7, cy]
            ], np.int32)
            cv2.fillPoly(img, [pts], color)
            cv2.fillPoly(mask, [pts], 255)
            segmentation = [[float(p) for point in pts for p in point]]
            x_min, y_min = pts.min(axis=0)
            x_max, y_max = pts.max(axis=0)
            bbox = [int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)]
            
        elif shape_name == "octagon":
            pts_list = generate_regular_polygon(cx, cy, size, 8, np.pi/8)
            pts = np.array([[pts_list[i], pts_list[i+1]] for i in range(0, len(pts_list), 2)], np.int32)
            cv2.fillPoly(img, [pts], color)
            cv2.fillPoly(mask, [pts], 255)
            segmentation = [pts_list]
            x_coords = pts[:, 0]
            y_coords = pts[:, 1]
            bbox = [int(x_coords.min()), int(y_coords.min()), 
                   int(x_coords.max() - x_coords.min()), int(y_coords.max() - y_coords.min())]
        
        # Save image and mask
        filename = f"single_{shape_name}.jpg"
        mask_filename = f"mask_{shape_name}.png"
        cv2.imwrite(str(images_dir / filename), img)
        cv2.imwrite(str(images_dir / mask_filename), mask)
        
        # Add to dataset
        coco_dataset["images"].append({
            "id": image_id,
            "width": 300,
            "height": 300,
            "file_name": filename
        })
        
        coco_dataset["annotations"].append({
            "id": annotation_id,
            "image_id": image_id,
            "category_id": cat_id,
            "bbox": [float(b) for b in bbox],
            "area": float(bbox[2] * bbox[3]),
            "segmentation": segmentation,
            "iscrowd": 0
        })
        
        image_id += 1
        annotation_id += 1
    
    # Save COCO annotations
    with open(output_path / "annotations.json", 'w') as f:
        json.dump(coco_dataset, f, indent=2)
    
    print(f"\nDataset created successfully!")
    print(f"- {len(coco_dataset['images'])} images")
    print(f"- {len(coco_dataset['annotations'])} annotations with segmentations")
    print(f"- {len(coco_dataset['categories'])} shape categories")
    print(f"Location: {output_path}")


if __name__ == "__main__":
    create_segmented_shapes_dataset()