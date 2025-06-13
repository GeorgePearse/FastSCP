"""COCO format annotation loader with caching for FastSCP."""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np
from pycocotools.coco import COCO


class COCOLoader:
    """Load and cache objects from COCO format annotations."""
    
    def __init__(self, annotation_file: str, image_dir: Optional[str] = None, 
                 cache_size: int = 1000):
        """
        Initialize the COCO loader.
        
        Args:
            annotation_file: Path to COCO format annotation JSON file
            image_dir: Directory containing images. If None, uses directory from annotation file
            cache_size: Maximum number of objects to cache in memory
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
        self._cache: Dict[int, np.ndarray] = {}  # annotation_id -> cropped object
        self._cache_order: List[int] = []  # LRU tracking
        
        # Load COCO annotations
        self.coco = COCO(str(self.annotation_file))
        self._load_metadata()
    
    def _load_metadata(self):
        """Load and organize metadata from COCO annotations."""
        self.category_ids = list(self.coco.cats.keys())
        self.categories = {cat['id']: cat['name'] for cat in self.coco.cats.values()}
        self.category_to_annotations: Dict[int, List[int]] = {}
        
        # Group annotations by category
        for ann_id, ann in self.coco.anns.items():
            cat_id = ann['category_id']
            if cat_id not in self.category_to_annotations:
                self.category_to_annotations[cat_id] = []
            self.category_to_annotations[cat_id].append(ann_id)
    
    def _update_cache(self, ann_id: int, crop: np.ndarray):
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
            
            self._cache[ann_id] = crop
            self._cache_order.append(ann_id)
    
    def get_category_id(self, category_name: str) -> Optional[int]:
        """Get category ID from category name."""
        for cat_id, name in self.categories.items():
            if name == category_name:
                return cat_id
        return None
    
    def load_object_crop(self, annotation_id: int) -> Optional[np.ndarray]:
        """
        Load a cropped object from an annotation.
        
        Args:
            annotation_id: COCO annotation ID
            
        Returns:
            Cropped object as numpy array or None if failed
        """
        # Check cache first
        if annotation_id in self._cache:
            self._update_cache(annotation_id, self._cache[annotation_id])
            return self._cache[annotation_id].copy()
        
        # Load annotation
        if annotation_id not in self.coco.anns:
            return None
        
        ann = self.coco.anns[annotation_id]
        
        # Load image
        img_info = self.coco.imgs[ann['image_id']]
        img_path = self.image_dir / img_info['file_name']
        
        if not img_path.exists():
            return None
        
        image = cv2.imread(str(img_path))
        if image is None:
            return None
        
        # Extract crop using bbox
        x, y, w, h = [int(v) for v in ann['bbox']]
        
        # Ensure bbox is within image bounds
        x = max(0, x)
        y = max(0, y)
        w = min(w, image.shape[1] - x)
        h = min(h, image.shape[0] - y)
        
        if w <= 0 or h <= 0:
            return None
        
        crop = image[y:y+h, x:x+w].copy()
        
        # Update cache
        self._update_cache(annotation_id, crop)
        
        return crop
    
    def get_random_objects(self, category: str, count: int) -> List[np.ndarray]:
        """
        Get random object crops from a specific category.
        
        Args:
            category: Category name (e.g., 'person', 'car')
            count: Number of objects to retrieve
            
        Returns:
            List of cropped objects as numpy arrays
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
            replace=count > len(available_annotations)
        )
        
        objects = []
        for ann_id in selected_ids:
            crop = self.load_object_crop(ann_id)
            if crop is not None:
                objects.append(crop)
        
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
            "hit_rate": 0.0  # Could track this with additional counters
        }