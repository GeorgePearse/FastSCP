# FastSCP Project Overview

## Mission Statement

Create a fast, efficient, and easy-to-use copy-paste augmentation library for computer vision that seamlessly integrates with existing ML pipelines.

## Project Goals

1. **Performance First**: Achieve sub-millisecond augmentation times for typical use cases
2. **Developer Friendly**: Simple, intuitive API that works out of the box
3. **Production Ready**: Robust error handling and comprehensive testing
4. **Community Driven**: Open source with clear contribution guidelines

## Target Users

- Machine Learning Engineers working on object detection/segmentation
- Researchers needing efficient data augmentation
- Teams requiring custom augmentation pipelines
- Anyone using Albumentations for computer vision tasks

## Key Features

- Simple dictionary-based configuration for object counts
- Runtime loading from COCO annotations
- Memory-efficient caching system
- Multiple blending modes
- Thread-safe operations
- Comprehensive test suite with synthetic data generation

## Success Metrics

- Augmentation speed: < 1ms per image (256x256)
- Memory efficiency: < 100MB for typical cache
- Code coverage: > 90%
- API simplicity: < 5 lines to get started

## Non-Goals

- Full COCO dataset management (use existing tools)
- Image annotation tools
- Model training utilities
- Complex geometric transformations (use Albumentations)

## Technical Constraints

- Python 3.8+ only
- Must maintain Albumentations compatibility
- Minimize external dependencies
- Cross-platform support (Linux, macOS, Windows)
