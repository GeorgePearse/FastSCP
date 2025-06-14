# FastSCP

Fast and efficient segmentation-based copy-paste augmentation for computer vision with precise object placement.

<img width="1339" alt="image" src="https://github.com/user-attachments/assets/b4d97f12-ebca-4587-a0ed-ce17d4ec0687" />

## Features

- 🚀 **Fast**: ~8ms per augmentation on average
- 🎯 **Precise**: Pixel-perfect segmentation masks for clean object extraction
- 🔄 **Rotation Support**: Rotate objects without rectangular artifacts
- 🎨 **Multiple Blend Modes**: Overlay, mix, and alpha blending with smooth edges
- 📦 **Albumentations Compatible**: Seamlessly integrates with existing pipelines
- 🗂️ **COCO Format**: Works with standard COCO annotation files with segmentation
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

## Key Features

| Feature | Description |
|---------|-------------|
| Shape Accuracy | Exact object boundaries |
| Background | Clean extraction without artifacts |
| Rotation | Full support without rectangular boundaries |
| Edge Quality | Smooth, anti-aliased edges |
| Performance | ~8ms per augmentation |

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

Your COCO annotations must include segmentation masks:

```json
{
    "images": [...],
    "annotations": [{
        "id": 1,
        "image_id": 1,
        "category_id": 1,
        "bbox": [x, y, width, height],
        "segmentation": [[x1, y1, x2, y2, ...]],  // Polygon points
        "area": 1234
    }],
    "categories": [...]
}
```

## Performance

Tested on MacBook Pro (Apple Silicon):
- Average: ~8ms per image
- Includes mask extraction, transformation, and blending
- Suitable for training-time augmentation

## Visualization Tools

FastSCP includes comprehensive visualization tools:

```bash
# Generate test visualizations
python test_precise_shapes.py

# Create HTML viewer
python visualize_precise_shapes.py
```

## Examples

Check out the `examples/` directory for:
- `demo.py` - Basic usage examples

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
│   ├── transforms_segmented.py  # Segmentation based transform
│   └── coco_loader_segmented.py # Segmentation-aware loader
├── tests/
│   ├── create_precise_shapes.py # Generate test data with masks
│   └── test_*.py               # Unit tests
└── examples/
    └── demo.py                 # Usage examples
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License

## Related Projects

- [copy-paste-aug](https://github.com/conradry/copy-paste-aug) - Another implementation of copy-paste augmentation for instance segmentation

## Acknowledgments

Built to work seamlessly with:
- [Albumentations](https://github.com/albumentations-team/albumentations)
- [COCO Dataset Format](https://cocodataset.org/)
- [pycocotools](https://github.com/cocodataset/cocoapi)
