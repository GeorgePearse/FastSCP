"""SimpleCopyPaste transformation for Albumentations."""

from typing import Dict, Optional, List, Any, Tuple
import random

import cv2
import numpy as np
from albumentations.core.transforms_interface import ImageOnlyTransform

from .coco_loader import COCOLoader


class SimpleCopyPaste(ImageOnlyTransform):
    """
    Simple copy-paste augmentation that loads objects from COCO annotations.
    
    This transform:
    1. Loads random objects from specified categories
    2. Pastes them onto the target image
    3. Uses simple overlay blending
    
    Args:
        coco_file: Path to COCO format annotation file
        object_counts: Dictionary mapping category names to counts
                      e.g., {"person": 2, "car": 1}
        image_dir: Directory containing source images (optional)
        blend_mode: Blending mode ('overlay', 'mix')
        scale_range: Range for random scaling of pasted objects
        min_visibility: Minimum visibility ratio for pasted objects
        always_apply: If True, always apply transform
        p: Probability of applying transform
    """
    
    def __init__(
        self,
        coco_file: str,
        object_counts: Dict[str, int],
        image_dir: Optional[str] = None,
        blend_mode: str = "overlay",
        scale_range: Tuple[float, float] = (0.5, 1.5),
        min_visibility: float = 0.3,
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super().__init__(always_apply=always_apply, p=p)
        
        self.coco_loader = COCOLoader(coco_file, image_dir)
        self.object_counts = object_counts
        self.blend_mode = blend_mode
        self.scale_range = scale_range
        self.min_visibility = min_visibility
        
        # Validate categories
        available_categories = set(self.coco_loader.get_all_categories().values())
        for category in object_counts.keys():
            if category not in available_categories:
                raise ValueError(f"Category '{category}' not found in COCO annotations. "
                               f"Available categories: {available_categories}")
    
    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        """
        Apply copy-paste augmentation to the image.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Augmented image
        """
        result = image.copy()
        h, w = image.shape[:2]
        
        # Collect all objects to paste
        objects_to_paste = []
        for category, count in self.object_counts.items():
            if count > 0:
                objects = self.coco_loader.get_random_objects(category, count)
                objects_to_paste.extend([(obj, category) for obj in objects])
        
        # Randomly shuffle to mix categories
        random.shuffle(objects_to_paste)
        
        # Paste each object
        for obj_img, category in objects_to_paste:
            result = self._paste_object(result, obj_img, (h, w))
        
        return result
    
    def _paste_object(self, image: np.ndarray, obj_img: np.ndarray, 
                     target_shape: Tuple[int, int]) -> np.ndarray:
        """
        Paste a single object onto the image.
        
        Args:
            image: Target image
            obj_img: Object to paste
            target_shape: Shape of target image (h, w)
            
        Returns:
            Image with object pasted
        """
        h, w = target_shape
        obj_h, obj_w = obj_img.shape[:2]
        
        # Random scale
        scale = random.uniform(*self.scale_range)
        new_h = int(obj_h * scale)
        new_w = int(obj_w * scale)
        
        # Skip if too small
        if new_h < 10 or new_w < 10:
            return image
        
        # Resize object
        obj_resized = cv2.resize(obj_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Random position (ensure minimum visibility)
        max_x = w - int(new_w * self.min_visibility)
        max_y = h - int(new_h * self.min_visibility)
        min_x = -int(new_w * (1 - self.min_visibility))
        min_y = -int(new_h * (1 - self.min_visibility))
        
        if max_x <= min_x or max_y <= min_y:
            return image
        
        x = random.randint(min_x, max_x)
        y = random.randint(min_y, max_y)
        
        # Calculate visible region
        obj_x1 = max(0, -x)
        obj_y1 = max(0, -y)
        obj_x2 = min(new_w, w - x)
        obj_y2 = min(new_h, h - y)
        
        img_x1 = max(0, x)
        img_y1 = max(0, y)
        img_x2 = min(w, x + new_w)
        img_y2 = min(h, y + new_h)
        
        # Skip if no visible region
        if obj_x2 <= obj_x1 or obj_y2 <= obj_y1:
            return image
        
        # Extract regions
        obj_region = obj_resized[obj_y1:obj_y2, obj_x1:obj_x2]
        img_region = image[img_y1:img_y2, img_x1:img_x2]
        
        # Apply blending
        blended = self._blend(img_region, obj_region)
        
        # Place back
        result = image.copy()
        result[img_y1:img_y2, img_x1:img_x2] = blended
        
        return result
    
    def _blend(self, background: np.ndarray, foreground: np.ndarray) -> np.ndarray:
        """
        Blend foreground object with background.
        
        Args:
            background: Background image region
            foreground: Foreground object to blend
            
        Returns:
            Blended image region
        """
        if self.blend_mode == "overlay":
            # Simple overlay - just replace
            return foreground
        
        elif self.blend_mode == "mix":
            # Mix with random alpha
            alpha = random.uniform(0.7, 1.0)
            return cv2.addWeighted(background, 1 - alpha, foreground, alpha, 0)
        
        else:
            return foreground
    
    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        """Get transform initialization argument names."""
        return (
            "coco_file",
            "object_counts", 
            "image_dir",
            "blend_mode",
            "scale_range",
            "min_visibility",
        )