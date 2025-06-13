"""Create shapes with exact masks and COCO annotations from scratch."""

import json
import os
from pathlib import Path
from datetime import datetime
import cv2
import numpy as np
from typing import List, Tuple, Dict, Any


class PreciseShapeGenerator:
    """Generate shapes with exact masks and COCO annotations."""
    
    def __init__(self, output_dir: str = "tests/precise_shapes"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir = self.output_dir / "images"
        self.masks_dir = self.output_dir / "masks"
        self.images_dir.mkdir(exist_ok=True)
        self.masks_dir.mkdir(exist_ok=True)
        
        # Define colors (BGR format for OpenCV)
        self.colors = {
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
        self.coco_dataset = {
            "info": {
                "description": "Precise Shapes Dataset with Exact Masks",
                "version": "1.0",
                "year": 2025,
                "contributor": "FastSCP",
                "date_created": datetime.now().isoformat()
            },
            "licenses": [{"id": 1, "name": "Test License", "url": ""}],
            "images": [],
            "annotations": [],
            "categories": []
        }
        
        self.annotation_id = 1
        self.image_id = 1
    
    def create_shape_on_canvas(self, canvas_size: Tuple[int, int], 
                              shape_type: str, position: Tuple[int, int], 
                              size: int, color: Tuple[int, int, int]) -> Tuple[np.ndarray, np.ndarray, List[float]]:
        """
        Create a shape on a canvas with its exact mask.
        
        Returns:
            - image: The image with the shape
            - mask: Binary mask of the shape
            - segmentation: COCO polygon segmentation
        """
        h, w = canvas_size
        image = np.zeros((h, w, 3), dtype=np.uint8)
        mask = np.zeros((h, w), dtype=np.uint8)
        cx, cy = position
        
        if shape_type == "square":
            half_size = size // 2
            pts = np.array([
                [cx - half_size, cy - half_size],
                [cx + half_size, cy - half_size],
                [cx + half_size, cy + half_size],
                [cx - half_size, cy + half_size]
            ], np.int32)
            cv2.fillPoly(image, [pts], color)
            cv2.fillPoly(mask, [pts], 255)
            segmentation = [float(coord) for point in pts for coord in point]
            
        elif shape_type == "circle":
            cv2.circle(image, (cx, cy), size, color, -1)
            cv2.circle(mask, (cx, cy), size, 255, -1)
            # Create polygon approximation of circle
            num_points = 32
            pts = []
            for i in range(num_points):
                angle = 2 * np.pi * i / num_points
                x = cx + size * np.cos(angle)
                y = cy + size * np.sin(angle)
                pts.extend([x, y])
            segmentation = pts
            
        elif shape_type == "triangle":
            pts = np.array([
                [cx, cy - size],
                [cx - int(size * 0.866), cy + size // 2],
                [cx + int(size * 0.866), cy + size // 2]
            ], np.int32)
            cv2.fillPoly(image, [pts], color)
            cv2.fillPoly(mask, [pts], 255)
            segmentation = [float(coord) for point in pts for coord in point]
            
        elif shape_type == "diamond":
            pts = np.array([
                [cx, cy - size],
                [cx + int(size * 0.7), cy],
                [cx, cy + size],
                [cx - int(size * 0.7), cy]
            ], np.int32)
            cv2.fillPoly(image, [pts], color)
            cv2.fillPoly(mask, [pts], 255)
            segmentation = [float(coord) for point in pts for coord in point]
            
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
            cv2.fillPoly(image, [pts], color)
            cv2.fillPoly(mask, [pts], 255)
            segmentation = [float(coord) for point in star_pts for coord in point]
            
        elif shape_type == "hexagon":
            pts = []
            for i in range(6):
                angle = 2 * np.pi * i / 6
                x = cx + size * np.cos(angle)
                y = cy + size * np.sin(angle)
                pts.append([x, y])
            pts = np.array(pts, np.int32)
            cv2.fillPoly(image, [pts], color)
            cv2.fillPoly(mask, [pts], 255)
            segmentation = [float(coord) for point in pts for coord in point]
        
        else:
            raise ValueError(f"Unknown shape type: {shape_type}")
        
        return image, mask, segmentation
    
    def create_individual_shapes(self):
        """Create individual shape images with white backgrounds."""
        print("Creating individual shapes with precise masks...")
        
        shapes = [
            ("red_square", "square", "red"),
            ("green_circle", "circle", "green"),
            ("blue_triangle", "triangle", "blue"),
            ("yellow_diamond", "diamond", "yellow"),
            ("purple_star", "star", "purple"),
            ("cyan_hexagon", "hexagon", "cyan"),
        ]
        
        # Add categories
        for i, (name, _, _) in enumerate(shapes, 1):
            self.coco_dataset["categories"].append({
                "id": i,
                "name": name,
                "supercategory": "shape"
            })
        
        # Create each shape
        for cat_id, (name, shape_type, color_name) in enumerate(shapes, 1):
            # Create shape on transparent background
            shape_img, shape_mask, segmentation = self.create_shape_on_canvas(
                (300, 300), shape_type, (150, 150), 80, self.colors[color_name]
            )
            
            # Create final image with white background
            white_bg = np.ones((300, 300, 3), dtype=np.uint8) * 255
            mask_3ch = cv2.cvtColor(shape_mask, cv2.COLOR_GRAY2BGR) / 255.0
            final_img = (shape_img * mask_3ch + white_bg * (1 - mask_3ch)).astype(np.uint8)
            
            # Save image and mask
            img_filename = f"shape_{name}.jpg"
            mask_filename = f"mask_{name}.png"
            cv2.imwrite(str(self.images_dir / img_filename), final_img)
            cv2.imwrite(str(self.masks_dir / mask_filename), shape_mask)
            
            # Save shape only (for copy-paste)
            shape_only = np.zeros((300, 300, 4), dtype=np.uint8)
            shape_only[:, :, :3] = shape_img
            shape_only[:, :, 3] = shape_mask
            cv2.imwrite(str(self.images_dir / f"shape_only_{name}.png"), shape_only)
            
            # Add to COCO dataset
            self.coco_dataset["images"].append({
                "id": self.image_id,
                "width": 300,
                "height": 300,
                "file_name": img_filename
            })
            
            # Calculate bbox from mask
            coords = np.column_stack(np.where(shape_mask > 0))
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            bbox = [float(x_min), float(y_min), float(x_max - x_min + 1), float(y_max - y_min + 1)]
            
            self.coco_dataset["annotations"].append({
                "id": self.annotation_id,
                "image_id": self.image_id,
                "category_id": cat_id,
                "bbox": bbox,
                "area": float(np.sum(shape_mask > 0)),
                "segmentation": [segmentation],
                "iscrowd": 0
            })
            
            self.annotation_id += 1
            self.image_id += 1
            
            print(f"  Created {name} with exact mask")
    
    def create_composite_scenes(self):
        """Create scenes with multiple shapes."""
        print("\nCreating composite scenes...")
        
        backgrounds = [
            ("white", np.ones((600, 800, 3), dtype=np.uint8) * 255),
            ("gray", np.ones((600, 800, 3), dtype=np.uint8) * 180),
            ("black", np.zeros((600, 800, 3), dtype=np.uint8)),
        ]
        
        for bg_idx, (bg_name, background) in enumerate(backgrounds):
            # Create 2 scenes per background
            for scene_idx in range(2):
                scene_img = background.copy()
                scene_mask = np.zeros((600, 800), dtype=np.uint8)
                
                filename = f"scene_{bg_name}_{scene_idx:02d}.jpg"
                
                # Add to COCO dataset
                self.coco_dataset["images"].append({
                    "id": self.image_id,
                    "width": 800,
                    "height": 600,
                    "file_name": filename
                })
                
                # Add random shapes
                shape_configs = [
                    ("square", (200, 150), 60, "red", 1),
                    ("circle", (600, 150), 50, "green", 2),
                    ("triangle", (400, 300), 70, "blue", 3),
                    ("star", (200, 450), 55, "purple", 5),
                    ("hexagon", (600, 450), 45, "cyan", 6),
                ]
                
                # Randomly select 3-4 shapes
                num_shapes = np.random.randint(3, 5)
                selected_shapes = np.random.choice(len(shape_configs), num_shapes, replace=False)
                
                for shape_idx in selected_shapes:
                    shape_type, pos, size, color, cat_id = shape_configs[shape_idx]
                    
                    # Create shape
                    shape_img, shape_mask, segmentation = self.create_shape_on_canvas(
                        (600, 800), shape_type, pos, size, self.colors[color]
                    )
                    
                    # Composite onto scene
                    mask_bool = shape_mask > 0
                    scene_img[mask_bool] = shape_img[mask_bool]
                    scene_mask[mask_bool] = 255
                    
                    # Calculate bbox
                    coords = np.column_stack(np.where(shape_mask > 0))
                    if len(coords) > 0:
                        y_min, x_min = coords.min(axis=0)
                        y_max, x_max = coords.max(axis=0)
                        bbox = [float(x_min), float(y_min), 
                               float(x_max - x_min + 1), float(y_max - y_min + 1)]
                        
                        # Add annotation
                        self.coco_dataset["annotations"].append({
                            "id": self.annotation_id,
                            "image_id": self.image_id,
                            "category_id": cat_id,
                            "bbox": bbox,
                            "area": float(np.sum(shape_mask > 0)),
                            "segmentation": [segmentation],
                            "iscrowd": 0
                        })
                        self.annotation_id += 1
                
                # Save scene
                cv2.imwrite(str(self.images_dir / filename), scene_img)
                cv2.imwrite(str(self.masks_dir / f"mask_{filename[:-4]}.png"), scene_mask)
                
                self.image_id += 1
                print(f"  Created {filename}")
    
    def save_annotations(self):
        """Save COCO annotations."""
        with open(self.output_dir / "annotations.json", 'w') as f:
            json.dump(self.coco_dataset, f, indent=2)
        
        print(f"\nDataset created successfully!")
        print(f"- {len(self.coco_dataset['images'])} images")
        print(f"- {len(self.coco_dataset['annotations'])} annotations with precise segmentations")
        print(f"- {len(self.coco_dataset['categories'])} categories")
        print(f"Location: {self.output_dir}")
    
    def generate(self):
        """Generate the complete dataset."""
        self.create_individual_shapes()
        self.create_composite_scenes()
        self.save_annotations()


def main():
    """Generate precise shapes dataset."""
    generator = PreciseShapeGenerator()
    generator.generate()


if __name__ == "__main__":
    main()