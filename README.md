# FastSCP

Fast and efficient copy-paste augmentation for computer vision, with support for both bounding box and precise segmentation-based object placement.

<img width="1339" alt="image" src="https://github.com/user-attachments/assets/b4d97f12-ebca-4587-a0ed-ce17d4ec0687" />

## Features

- 🚀 **Fast**: Sub-millisecond augmentation times (avg ~0.3ms for bbox, ~8ms for segmentation)
- 🎯 **Precise**: Supports both traditional bounding box and pixel-perfect segmentation masks
- 🔄 **Rotation Support**: Rotate objects without rectangular artifacts (segmentation mode)
- 🎨 **Multiple Blend Modes**: Overlay, mix, and alpha blending with smooth edges
- 📦 **Albumentations Compatible**: Seamlessly integrates with existing pipelines
- 🗂️ **COCO Format**: Works with standard COCO annotation files
- ⚡ **Smart Caching**: LRU cache for efficient object reuse

## Installation

```bash
pip install -e .
```

### Dependencies
- numpy >= 1.20.0
- opencv-python >= 4.5.0
- albumentations >= 1.3.0
- pycocotools >= 2.0.0

## Quick Start

### Basic Usage (Bounding Box Mode)

```python
from fastscp import SimpleCopyPaste

# Create transform
transform = SimpleCopyPaste(
    coco_file="path/to/annotations.json",
    object_counts={"person": 2, "car": 1},  # Paste 2 people and 1 car
    scale_range=(0.5, 1.5),                 # Random scaling
    p=0.5                                   # 50% probability
)

# Apply to image
augmented = transform(image=image)["image"]
```

### Advanced Usage (Segmentation Mode)

For precise object boundaries without background artifacts:

```python
from fastscp import SimpleCopyPasteSegmented

# Create transform with segmentation support
transform = SimpleCopyPasteSegmented(
    coco_file="path/to/annotations_with_segmentation.json",
    object_counts={"cat": 2, "dog": 1},
    rotation_range=45,      # ±45 degrees rotation
    blend_mode="alpha",     # Smooth edge blending
    scale_range=(0.5, 2.0),
    p=1.0
)

# Apply to image
augmented = transform(image=image)["image"]
```

## Key Differences: BBox vs Segmentation

| Feature | Bounding Box | Segmentation |
|---------|--------------|--------------|
| Speed | ~0.3ms | ~8ms |
| Shape Accuracy | Rectangular crops | Exact object boundaries |
| Background Artifacts | Includes background | Clean extraction |
| Rotation Support | Limited | Full support without artifacts |
| Edge Quality | Hard edges | Smooth, anti-aliased |
| Use Case | Fast augmentation | High-quality augmentation |

## Integration with Albumentations

```python
import albumentations as A
from fastscp import SimpleCopyPasteSegmented

pipeline = A.Compose([
    A.RandomBrightnessContrast(p=0.2),
    SimpleCopyPasteSegmented(
        coco_file="annotations.json",
        object_counts={"object": 3},
        rotation_range=30,
        p=0.8
    ),
    A.HorizontalFlip(p=0.5),
])

augmented = pipeline(image=image)
```

## Dataset Preparation

### Creating COCO Annotations with Segmentation

FastSCP includes utilities to generate test datasets with precise segmentation masks:

```python
from tests.create_precise_shapes import PreciseShapeGenerator

# Generate shapes with exact masks
generator = PreciseShapeGenerator(output_dir="my_dataset")
generator.generate()
```

This creates:
- Individual shape images with masks
- COCO format annotations with polygon segmentation
- Composite scenes for testing

### COCO Format Requirements

For bounding box mode:
```json
{
    "images": [...],
    "annotations": [{
        "id": 1,
        "image_id": 1,
        "category_id": 1,
        "bbox": [x, y, width, height],
        "area": 1234
    }],
    "categories": [...]
}
```

For segmentation mode, add:
```json
{
    "annotations": [{
        ...
        "segmentation": [[x1, y1, x2, y2, ...]],  // Polygon points
        ...
    }]
}
```

## Performance

Tested on MacBook Pro (Apple Silicon):

- **SimpleCopyPaste (BBox)**: 
  - Average: 0.3ms per image
  - Suitable for real-time augmentation
  
- **SimpleCopyPasteSegmented**: 
  - Average: 8ms per image
  - Higher quality results with precise boundaries

## Visualization Tools

FastSCP includes comprehensive visualization tools:

```bash
# Generate test visualizations
python test_precise_shapes.py

# Create HTML viewer
python visualize_precise_shapes.py

# Compare bbox vs segmentation
python compare_bbox_vs_segmentation.py
```

## Examples

Check out the `examples/` directory for:
- `demo.py` - Basic usage examples
- `test_transformations.py` - Comprehensive transformation tests
- `simple_test.py` - Quick before/after comparison

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=fastscp
```

## Project Structure

```
FastSCP/
├── fastscp/
│   ├── transforms.py            # Bounding box based transform
│   ├── transforms_segmented.py  # Segmentation based transform
│   ├── coco_loader.py          # Basic COCO loader
│   └── coco_loader_segmented.py # Segmentation-aware loader
├── tests/
│   ├── create_precise_shapes.py # Generate test data with masks
│   └── test_*.py               # Unit tests
└── examples/
    └── *.py                    # Usage examples
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License

## Acknowledgments

Built to work seamlessly with:
- [Albumentations](https://github.com/albumentations-team/albumentations)
- [COCO Dataset Format](https://cocodataset.org/)
- [pycocotools](https://github.com/cocodataset/cocoapi)
