# FastSCP Architecture

## Project Overview

FastSCP is a high-performance Python library for copy-paste augmentation in computer vision pipelines. It implements an Albumentations-compliant transformation that loads cropped objects from COCO-format annotations and pastes them into images at runtime.

## Core Requirements

- **Performance**: Must be fast for real-time augmentation during training
- **Compatibility**: Albumentations-compliant interface
- **Flexibility**: User-configurable object counts via dictionary
- **Runtime Loading**: Objects loaded from COCO annotations on-demand

## Technical Architecture

### Component Structure

```
FastSCP/
├── fastscp/
│   ├── __init__.py
│   ├── transforms.py      # Main SimpleCopyPaste transformation
│   ├── coco_loader.py     # COCO annotation and object loading
│   ├── cache.py           # Object caching for performance
│   └── utils.py           # Helper functions
├── tests/
│   ├── __init__.py
│   ├── test_transforms.py
│   ├── test_coco_loader.py
│   └── data/              # Test images and annotations
├── examples/
│   └── demo.py            # Usage examples
└── setup.py
```

### Key Design Decisions

1. **Caching Strategy**: Pre-load and cache cropped objects by category to avoid repeated file I/O
2. **Memory Management**: Configurable cache size with LRU eviction
3. **Blend Modes**: Support multiple blending strategies (overlay, mix, seamless)
4. **Coordinate Systems**: Handle both absolute and relative coordinates
5. **Batch Processing**: Optimize for batch operations when possible

### Performance Optimizations

- NumPy-based operations for speed
- Optional numba JIT compilation for critical paths
- Lazy loading of objects
- Efficient memory layout for cache
- Vectorized blending operations

### Integration Points

- **Albumentations**: Inherit from `DualTransform` or `ImageOnlyTransform`
- **COCO Format**: Use pycocotools for annotation parsing
- **OpenCV**: Primary image manipulation library
- **NumPy**: Core array operations

## Dependencies

- albumentations >= 1.3.0
- opencv-python >= 4.5.0
- numpy >= 1.20.0
- pycocotools >= 2.0.0
- (optional) numba >= 0.55.0 for performance

## Testing Strategy

- Unit tests for each component
- Integration tests with real COCO data
- Performance benchmarks
- Visual regression tests with generated test images
