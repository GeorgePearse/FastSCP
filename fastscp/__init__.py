"""FastSCP - Fast copy-paste augmentation for computer vision."""

__version__ = "0.1.0"

from .transforms import SimpleCopyPaste
from .coco_loader import COCOLoader

__all__ = [
    "SimpleCopyPaste",
    "COCOLoader",
]
