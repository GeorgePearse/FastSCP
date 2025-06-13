"""FastSCP - Fast copy-paste augmentation for computer vision."""

__version__ = "0.1.0"

from .transforms_segmented import SimpleCopyPasteSegmented
from .coco_loader_segmented import COCOLoaderSegmented

__all__ = [
    "SimpleCopyPasteSegmented",
    "COCOLoaderSegmented",
]
