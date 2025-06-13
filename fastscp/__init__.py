"""FastSCP - Fast copy-paste augmentation for computer vision."""

__version__ = "0.1.0"

from .transforms import SimpleCopyPaste
from .transforms_segmented import SimpleCopyPasteSegmented
from .coco_loader import COCOLoader
from .coco_loader_segmented import COCOLoaderSegmented

__all__ = ["SimpleCopyPaste", "SimpleCopyPasteSegmented", "COCOLoader", "COCOLoaderSegmented"]