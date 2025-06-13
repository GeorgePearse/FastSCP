"""COCO format annotation loader with segmentation mask support for precise cropping."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils


class COCOLoader:
    """Load and cache objects from COCO format annotations using segmentation masks."""

    def __init__(
        self,
        annotation_file: str,
        image_dir: Optional[str] = None,
        cache_size: int = 1000,
        use_bbox_fallback: bool = True,
    ):
        """
        Initialize the COCO loader with segmentation support.

        Args:
            annotation_file: Path to COCO format annotation JSON file
            image_dir: Directory containing images. If None, uses directory from annotation file
            cache_size: Maximum number of objects to cache in memory
            use_bbox_fallback: If True, falls back to bbox crop when segmentation is not available
        """
        self.annotation_file = Path(annotation_file)
        if not self.annotation_file.exists():
            raise FileNotFoundError(f"Annotation file not found: {annotation_file}")

        # Set image directory
        if image_dir is None:
            self.image_dir = self.annotation_file.parent / "images"
        else:
            self.image_dir = Path(image_dir)

        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")

        self.cache_size = cache_size
        self.use_bbox_fallback = use_bbox_fallback
        self._cache: Dict[
            int, Tuple[np.ndarray, np.ndarray]
        ] = {}  # annotation_id -> (cropped object, mask)
        self._cache_order: List[int] = []  # LRU tracking

        # Load COCO annotations
        self.coco = COCO(str(self.annotation_file))
        self._load_metadata()

    def _load_metadata(self):
        """Load and organize metadata from COCO annotations."""
        self.category_ids = list(self.coco.cats.keys())
        self.categories = {cat["id"]: cat["name"] for cat in self.coco.cats.values()}
        self.category_to_annotations: Dict[int, List[int]] = {}

        # Group annotations by category
        for ann_id, ann in self.coco.anns.items():
            cat_id = ann["category_id"]
            if cat_id not in self.category_to_annotations:
                self.category_to_annotations[cat_id] = []
            self.category_to_annotations[cat_id].append(ann_id)

    def _update_cache(self, ann_id: int, crop: np.ndarray, mask: np.ndarray):
        """Update cache with LRU eviction."""
        if ann_id in self._cache:
            # Move to end (most recently used)
            self._cache_order.remove(ann_id)
            self._cache_order.append(ann_id)
        else:
            # Add new item
            if len(self._cache) >= self.cache_size:
                # Evict least recently used
                evict_id = self._cache_order.pop(0)
                del self._cache[evict_id]

            self._cache[ann_id] = (crop, mask)
            self._cache_order.append(ann_id)

    def get_category_id(self, category_name: str) -> Optional[int]:
        """Get category ID from category name."""
        for cat_id, name in self.categories.items():
            if name == category_name:
                return cat_id
        return None

    def _create_mask_from_segmentation(
        self, ann: Dict[str, Any], img_shape: Tuple[int, int]
    ) -> np.ndarray:
        """Create binary mask from segmentation annotation."""
        h, w = img_shape[:2]

        if "segmentation" in ann and ann["segmentation"]:
            # Handle different segmentation formats
            if isinstance(ann["segmentation"], dict):
                # RLE format
                if "counts" in ann["segmentation"]:
                    rle = ann["segmentation"]
                    mask = maskUtils.decode(rle)
                else:
                    # Invalid format, return empty mask
                    return np.zeros((h, w), dtype=np.uint8)
            elif isinstance(ann["segmentation"], list):
                # Polygon format
                mask = np.zeros((h, w), dtype=np.uint8)
                for seg in ann["segmentation"]:
                    if len(seg) >= 6:  # Need at least 3 points
                        # Convert to points
                        pts = np.array(seg).reshape(-1, 2).astype(np.int32)
                        cv2.fillPoly(mask, [pts], 255)
                return mask
            else:
                return np.zeros((h, w), dtype=np.uint8)
        else:
            return np.zeros((h, w), dtype=np.uint8)

        return mask

    def load_object_crop(
        self, annotation_id: int, include_context: float = 0.0
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Load a cropped object from an annotation using segmentation mask.

        Args:
            annotation_id: COCO annotation ID
            include_context: Fraction of bbox size to include as context (0.0 = tight crop)

        Returns:
            Tuple of (cropped object with transparent background, binary mask) or None if failed
        """
        # Check cache first
        if annotation_id in self._cache:
            self._update_cache(annotation_id, *self._cache[annotation_id])
            crop, mask = self._cache[annotation_id]
            return crop.copy(), mask.copy()

        # Load annotation
        if annotation_id not in self.coco.anns:
            return None

        ann = self.coco.anns[annotation_id]

        # Load image
        img_info = self.coco.imgs[ann["image_id"]]
        img_path = self.image_dir / img_info["file_name"]

        if not img_path.exists():
            return None

        image = cv2.imread(str(img_path))
        if image is None:
            return None

        # Create mask from segmentation
        mask = self._create_mask_from_segmentation(ann, image.shape)

        # If no segmentation or mask is empty, fall back to bbox
        if mask.sum() == 0 and self.use_bbox_fallback:
            x, y, w, h = [int(v) for v in ann["bbox"]]
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            mask[y : y + h, x : x + w] = 255

        # Find bounding box of mask
        coords = np.column_stack(np.where(mask > 0))
        if len(coords) == 0:
            return None

        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)

        # Add context if requested
        if include_context > 0:
            h, w = image.shape[:2]
            context_h = int((y_max - y_min) * include_context)
            context_w = int((x_max - x_min) * include_context)

            y_min = max(0, y_min - context_h)
            y_max = min(h, y_max + context_h)
            x_min = max(0, x_min - context_w)
            x_max = min(w, x_max + context_w)

        # Crop image and mask
        crop = image[y_min : y_max + 1, x_min : x_max + 1].copy()
        mask_crop = mask[y_min : y_max + 1, x_min : x_max + 1].copy()

        # Create RGBA image with transparency
        if crop.shape[2] == 3:
            # Add alpha channel
            rgba = np.zeros((crop.shape[0], crop.shape[1], 4), dtype=np.uint8)
            rgba[:, :, :3] = crop
            rgba[:, :, 3] = mask_crop
            crop = rgba

        # Update cache
        self._update_cache(annotation_id, crop, mask_crop)

        return crop, mask_crop

    def get_random_objects(
        self, category: str, count: int, include_context: float = 0.0
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Get random object crops from a specific category.

        Args:
            category: Category name (e.g., 'person', 'car')
            count: Number of objects to retrieve
            include_context: Fraction of bbox size to include as context

        Returns:
            List of tuples (cropped object with alpha channel, binary mask)
        """
        cat_id = self.get_category_id(category)
        if cat_id is None:
            return []

        if cat_id not in self.category_to_annotations:
            return []

        available_annotations = self.category_to_annotations[cat_id]
        if not available_annotations:
            return []

        # Sample with replacement if needed
        selected_ids = np.random.choice(
            available_annotations,
            size=min(count, len(available_annotations)),
            replace=count > len(available_annotations),
        )

        objects = []
        for ann_id in selected_ids:
            result = self.load_object_crop(ann_id, include_context)
            if result is not None:
                objects.append(result)

        return objects

    def get_all_categories(self) -> Dict[int, str]:
        """Get all available categories."""
        return self.categories.copy()

    def get_annotation_count(self, category: Optional[str] = None) -> int:
        """Get total number of annotations, optionally filtered by category."""
        if category is None:
            return len(self.coco.anns)

        cat_id = self.get_category_id(category)
        if cat_id is None or cat_id not in self.category_to_annotations:
            return 0

        return len(self.category_to_annotations[cat_id])

    def clear_cache(self):
        """Clear the object cache."""
        self._cache.clear()
        self._cache_order.clear()

    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about cache status."""
        return {
            "size": len(self._cache),
            "max_size": self.cache_size,
            "hit_rate": 0.0,  # Could track this with additional counters
        }
