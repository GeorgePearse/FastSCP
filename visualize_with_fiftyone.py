"""Generate transformed images and visualize them in FiftyOne."""

import json
import os
from pathlib import Path
from datetime import datetime
import time

import cv2
import numpy as np
import fiftyone as fo
import fiftyone.core.metadata as fom
from pycocotools.coco import COCO

from fastscp import SimpleCopyPaste


def generate_transformed_dataset(base_dir="tests/colored_shapes", output_dir="fiftyone_output", num_samples=20):
    """Generate a dataset with transformed images."""
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    images_dir = output_path / "images"
    images_dir.mkdir(exist_ok=True)
    
    # Load base images
    base_images = {
        "gray": cv2.imread(str(Path(base_dir) / "images" / "background_gray.jpg")),
        "gradient": cv2.imread(str(Path(base_dir) / "images" / "background_gradient.jpg")),
        "grid": cv2.imread(str(Path(base_dir) / "images" / "background_grid.jpg"))
    }
    
    # Different transformation configurations
    configs = [
        {
            "name": "basic_overlay",
            "object_counts": {"red_square": 2, "green_circle": 1},
            "blend_mode": "overlay",
            "scale_range": (0.8, 1.2)
        },
        {
            "name": "mix_blend",
            "object_counts": {"blue_triangle": 2, "yellow_rectangle": 1},
            "blend_mode": "mix",
            "scale_range": (0.5, 1.5)
        },
        {
            "name": "many_small",
            "object_counts": {"red_square": 3, "green_circle": 3},
            "blend_mode": "overlay",
            "scale_range": (0.3, 0.6)
        },
        {
            "name": "few_large",
            "object_counts": {"purple_star": 1, "cyan_hexagon": 1},
            "blend_mode": "overlay",
            "scale_range": (1.5, 2.0)
        },
        {
            "name": "mixed_all",
            "object_counts": {
                "red_square": 1,
                "green_circle": 1,
                "blue_triangle": 1,
                "yellow_rectangle": 1
            },
            "blend_mode": "mix",
            "scale_range": (0.5, 1.5)
        }
    ]
    
    # COCO-style dataset for FiftyOne
    coco_dataset = {
        "info": {
            "description": "FastSCP Transformed Dataset for FiftyOne",
            "version": "1.0",
            "year": 2025,
            "contributor": "FastSCP",
            "date_created": datetime.now().isoformat()
        },
        "licenses": [{"id": 1, "name": "Test License", "url": ""}],
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 1, "name": "pasted_object", "supercategory": "object"},
            {"id": 2, "name": "red_square", "supercategory": "shape"},
            {"id": 3, "name": "green_circle", "supercategory": "shape"},
            {"id": 4, "name": "blue_triangle", "supercategory": "shape"},
            {"id": 5, "name": "yellow_rectangle", "supercategory": "shape"},
            {"id": 6, "name": "purple_star", "supercategory": "shape"},
            {"id": 7, "name": "cyan_hexagon", "supercategory": "shape"},
        ]
    }
    
    image_id = 1
    annotation_id = 1
    
    print(f"Generating {num_samples} transformed images...")
    
    for i in range(num_samples):
        # Select configuration and background
        config = configs[i % len(configs)]
        bg_name = list(base_images.keys())[i % len(base_images)]
        bg_image = base_images[bg_name].copy()
        
        # Create transform
        transform = SimpleCopyPaste(
            coco_file=str(Path(base_dir) / "annotations.json"),
            object_counts=config["object_counts"],
            blend_mode=config["blend_mode"],
            scale_range=config["scale_range"],
            p=1.0
        )
        
        # Apply transformation
        start_time = time.time()
        augmented = transform(image=bg_image)["image"]
        transform_time = time.time() - start_time
        
        # Save image
        filename = f"transformed_{image_id:04d}_{config['name']}_{bg_name}.jpg"
        filepath = images_dir / filename
        cv2.imwrite(str(filepath), augmented)
        
        # Add image metadata
        h, w = augmented.shape[:2]
        coco_dataset["images"].append({
            "id": image_id,
            "width": w,
            "height": h,
            "file_name": filename,
            "date_captured": datetime.now().isoformat(),
            "transform_config": config["name"],
            "background": bg_name,
            "transform_time_ms": transform_time * 1000
        })
        
        # Add pseudo-annotations for visualization
        # (In real use, you'd track actual object locations)
        for obj_type, count in config["object_counts"].items():
            for j in range(count):
                # Generate random bbox for visualization
                # In production, you'd track actual pasted locations
                bbox_w = np.random.randint(50, 150)
                bbox_h = np.random.randint(50, 150)
                bbox_x = np.random.randint(0, w - bbox_w)
                bbox_y = np.random.randint(0, h - bbox_h)
                
                # Map object type to category
                cat_map = {
                    "red_square": 2,
                    "green_circle": 3,
                    "blue_triangle": 4,
                    "yellow_rectangle": 5,
                    "purple_star": 6,
                    "cyan_hexagon": 7
                }
                
                coco_dataset["annotations"].append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": cat_map.get(obj_type, 1),
                    "bbox": [bbox_x, bbox_y, bbox_w, bbox_h],
                    "area": bbox_w * bbox_h,
                    "segmentation": [],
                    "iscrowd": 0,
                    "attributes": {
                        "blend_mode": config["blend_mode"],
                        "scale_range": config["scale_range"],
                        "object_type": obj_type
                    }
                })
                annotation_id += 1
        
        image_id += 1
        
        if (i + 1) % 5 == 0:
            print(f"  Generated {i + 1}/{num_samples} images...")
    
    # Save COCO annotations
    annotations_path = output_path / "annotations.json"
    with open(annotations_path, 'w') as f:
        json.dump(coco_dataset, f, indent=2)
    
    print(f"\nDataset generated:")
    print(f"  - {len(coco_dataset['images'])} images")
    print(f"  - {len(coco_dataset['annotations'])} annotations")
    print(f"  - Saved to: {output_path}")
    
    return str(output_path)


def load_into_fiftyone(dataset_dir, dataset_name="fastscp_transformations"):
    """Load the generated dataset into FiftyOne."""
    
    print(f"\nLoading dataset into FiftyOne...")
    
    # Delete existing dataset if it exists
    if dataset_name in fo.list_datasets():
        fo.delete_dataset(dataset_name)
    
    # Create dataset
    dataset = fo.Dataset(dataset_name)
    dataset.persistent = True
    
    # Load COCO annotations
    coco_path = Path(dataset_dir) / "annotations.json"
    with open(coco_path) as f:
        coco_data = json.load(f)
    
    coco = COCO(str(coco_path))
    
    # Create category mapping
    categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
    
    # Add samples
    for img_data in coco_data['images']:
        # Create sample
        filepath = Path(dataset_dir) / "images" / img_data['file_name']
        sample = fo.Sample(filepath=str(filepath))
        
        # Add metadata
        sample["transform_config"] = img_data.get("transform_config", "unknown")
        sample["background"] = img_data.get("background", "unknown")
        sample["transform_time_ms"] = img_data.get("transform_time_ms", 0)
        
        # Get annotations for this image
        ann_ids = coco.getAnnIds(imgIds=img_data['id'])
        anns = coco.loadAnns(ann_ids)
        
        # Convert to FiftyOne detections
        detections = []
        for ann in anns:
            # Convert COCO bbox to relative coordinates
            x, y, w, h = ann['bbox']
            rel_x = x / img_data['width']
            rel_y = y / img_data['height']
            rel_w = w / img_data['width']
            rel_h = h / img_data['height']
            
            # Create detection
            label = categories.get(ann['category_id'], 'unknown')
            detection = fo.Detection(
                label=label,
                bounding_box=[rel_x, rel_y, rel_w, rel_h],
                confidence=1.0
            )
            
            # Add attributes if available
            if 'attributes' in ann:
                for key, value in ann['attributes'].items():
                    detection[key] = str(value)
            
            detections.append(detection)
        
        sample["ground_truth"] = fo.Detections(detections=detections)
        
        # Add sample to dataset
        dataset.add_sample(sample)
    
    print(f"  Added {len(dataset)} samples to dataset '{dataset_name}'")
    
    # Add dataset info
    dataset.info = {
        "description": "FastSCP transformation examples with synthetic annotations",
        "created": datetime.now().isoformat(),
        "transform_configs": [c["name"] for c in configs]
    }
    
    # Compute metadata
    dataset.compute_metadata()
    
    return dataset


def main():
    """Main function to generate and visualize dataset."""
    
    print("FastSCP FiftyOne Visualization")
    print("=" * 50)
    
    # Generate transformed dataset
    dataset_dir = generate_transformed_dataset(num_samples=30)
    
    # Load into FiftyOne
    dataset = load_into_fiftyone(dataset_dir)
    
    # Print statistics
    print("\nDataset Statistics:")
    print(f"  Total samples: {len(dataset)}")
    print(f"  Total detections: {dataset.count('ground_truth.detections')}")
    
    # Group by transform config
    print("\nSamples by transform config:")
    for config in dataset.distinct("transform_config"):
        count = len(dataset.match(fo.ViewField("transform_config") == config))
        print(f"  - {config}: {count} samples")
    
    # Launch FiftyOne app
    print("\nLaunching FiftyOne App...")
    print("Note: The bounding boxes are synthetic for visualization.")
    print("In production, you would track actual pasted object locations.\n")
    
    session = fo.launch_app(dataset)
    
    # Keep app running
    print("FiftyOne app is running at: http://localhost:5151")
    print("Press Ctrl+C to exit...")
    
    try:
        # Keep the script running
        session.wait()
    except KeyboardInterrupt:
        print("\nShutting down...")


if __name__ == "__main__":
    # Note: configs is defined globally for use in load_into_fiftyone
    configs = [
        {
            "name": "basic_overlay",
            "object_counts": {"red_square": 2, "green_circle": 1},
            "blend_mode": "overlay",
            "scale_range": (0.8, 1.2)
        },
        {
            "name": "mix_blend",
            "object_counts": {"blue_triangle": 2, "yellow_rectangle": 1},
            "blend_mode": "mix",
            "scale_range": (0.5, 1.5)
        },
        {
            "name": "many_small",
            "object_counts": {"red_square": 3, "green_circle": 3},
            "blend_mode": "overlay",
            "scale_range": (0.3, 0.6)
        },
        {
            "name": "few_large",
            "object_counts": {"purple_star": 1, "cyan_hexagon": 1},
            "blend_mode": "overlay",
            "scale_range": (1.5, 2.0)
        },
        {
            "name": "mixed_all",
            "object_counts": {
                "red_square": 1,
                "green_circle": 1,
                "blue_triangle": 1,
                "yellow_rectangle": 1
            },
            "blend_mode": "mix",
            "scale_range": (0.5, 1.5)
        }
    ]
    
    main()