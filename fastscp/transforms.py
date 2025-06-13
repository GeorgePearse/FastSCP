"""SimpleCopyPaste transformation with segmentation mask support for precise object placement."""

from typing import Dict, Optional, Tuple
import random

import cv2
import numpy as np
from albumentations.core.transforms_interface import ImageOnlyTransform

from .coco_loader import COCOLoader


class SimpleCopyPaste(ImageOnlyTransform):
    """
    Simple copy-paste augmentation that uses segmentation masks for precise object placement.

    This transform:
    1. Loads objects with their segmentation masks
    2. Pastes them using alpha blending for smooth edges
    3. Supports various blending modes

    Args:
        coco_file: Path to COCO format annotation file
        object_counts: Dictionary mapping category names to counts
                      e.g., {"person": 2, "car": 1}
        image_dir: Directory containing source images (optional)
        blend_mode: Blending mode ('overlay', 'mix', 'alpha')
        scale_range: Range for random scaling of pasted objects
        min_visibility: Minimum visibility ratio for pasted objects
        rotation_range: Range for random rotation in degrees (-angle, angle)
        include_context: Fraction of bbox to include as context (0.0 = tight crop)
        always_apply: If True, always apply transform
        p: Probability of applying transform
    """

    def __init__(
        self,
        coco_file: str,
        object_counts: Dict[str, int],
        image_dir: Optional[str] = None,
        blend_mode: str = "alpha",
        scale_range: Tuple[float, float] = (0.5, 1.5),
        min_visibility: float = 0.3,
        rotation_range: int = 0,
        include_context: float = 0.0,
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super().__init__(always_apply=always_apply, p=p)

        self.coco_loader = COCOLoader(coco_file, image_dir)
        self.object_counts = object_counts
        self.blend_mode = blend_mode
        self.scale_range = scale_range
        self.min_visibility = min_visibility
        self.rotation_range = rotation_range
        self.include_context = include_context

        # Validate categories
        available_categories = set(self.coco_loader.get_all_categories().values())
        for category in object_counts.keys():
            if category not in available_categories:
                raise ValueError(
                    f"Category '{category}' not found in COCO annotations. "
                    f"Available categories: {available_categories}"
                )

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
                objects = self.coco_loader.get_random_objects(
                    category, count, self.include_context
                )
                objects_to_paste.extend(
                    [(obj, mask, category) for obj, mask in objects]
                )

        # Randomly shuffle to mix categories
        random.shuffle(objects_to_paste)

        # Paste each object
        for obj_img, obj_mask, category in objects_to_paste:
            result = self._paste_object_with_mask(result, obj_img, obj_mask, (h, w))

        return result

    def _rotate_image_and_mask(
        self, image: np.ndarray, mask: np.ndarray, angle: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Rotate image and mask by given angle."""
        h, w = image.shape[:2]
        center = (w // 2, h // 2)

        # Get rotation matrix
        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Calculate new dimensions
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int(h * sin + w * cos)
        new_h = int(h * cos + w * sin)

        # Adjust rotation matrix
        M[0, 2] += (new_w - w) / 2
        M[1, 2] += (new_h - h) / 2

        # Rotate image and mask
        rotated_img = cv2.warpAffine(
            image,
            M,
            (new_w, new_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0),
        )
        rotated_mask = cv2.warpAffine(
            mask,
            M,
            (new_w, new_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        return rotated_img, rotated_mask

    def _paste_object_with_mask(
        self,
        image: np.ndarray,
        obj_img: np.ndarray,
        obj_mask: np.ndarray,
        target_shape: Tuple[int, int],
    ) -> np.ndarray:
        """
        Paste a single object onto the image using its mask.

        Args:
            image: Target image
            obj_img: Object image with alpha channel
            obj_mask: Binary mask for the object
            target_shape: Shape of target image (h, w)

        Returns:
            Image with object pasted
        """
        h, w = target_shape

        # Random rotation
        if self.rotation_range > 0:
            angle = random.uniform(-self.rotation_range, self.rotation_range)
            obj_img, obj_mask = self._rotate_image_and_mask(obj_img, obj_mask, angle)

        obj_h, obj_w = obj_img.shape[:2]

        # Random scale
        scale = random.uniform(*self.scale_range)
        new_h = int(obj_h * scale)
        new_w = int(obj_w * scale)

        # Skip if too small
        if new_h < 10 or new_w < 10:
            return image

        # Resize object and mask
        obj_resized = cv2.resize(
            obj_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR
        )
        mask_resized = cv2.resize(
            obj_mask, (new_w, new_h), interpolation=cv2.INTER_LINEAR
        )

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
        mask_region = mask_resized[obj_y1:obj_y2, obj_x1:obj_x2]
        img_region = image[img_y1:img_y2, img_x1:img_x2]

        # Apply blending
        blended = self._blend_with_mask(img_region, obj_region, mask_region)

        # Place back
        result = image.copy()
        result[img_y1:img_y2, img_x1:img_x2] = blended

        return result

    def _blend_with_mask(
        self, background: np.ndarray, foreground: np.ndarray, mask: np.ndarray
    ) -> np.ndarray:
        """
        Blend foreground object with background using mask.

        Args:
            background: Background image region
            foreground: Foreground object to blend (with or without alpha)
            mask: Binary mask for blending

        Returns:
            Blended image region
        """
        # Ensure mask is normalized
        if mask.max() > 1:
            mask = mask.astype(np.float32) / 255.0
        else:
            mask = mask.astype(np.float32)

        # Handle alpha channel if present
        if foreground.shape[2] == 4:
            # Extract alpha channel
            alpha = foreground[:, :, 3].astype(np.float32) / 255.0
            foreground_rgb = foreground[:, :, :3]
            # Combine mask with alpha
            mask = mask * alpha
        else:
            foreground_rgb = foreground

        # Ensure mask has right shape for broadcasting
        if len(mask.shape) == 2:
            mask = mask[:, :, np.newaxis]

        if self.blend_mode == "alpha":
            # Alpha blending with anti-aliasing
            # Apply slight Gaussian blur to mask edges for smoother blending
            if len(mask.shape) == 3:
                mask_smooth = cv2.GaussianBlur(mask[:, :, 0], (3, 3), 0.5)
            else:
                mask_smooth = cv2.GaussianBlur(mask, (3, 3), 0.5)
            mask_smooth = mask_smooth[:, :, np.newaxis]
            blended = (
                foreground_rgb * mask_smooth + background * (1 - mask_smooth)
            ).astype(np.uint8)

        elif self.blend_mode == "overlay":
            # Simple overlay - use mask as hard cutout
            mask_binary = (mask > 0.5).astype(np.float32)
            blended = (
                foreground_rgb * mask_binary + background * (1 - mask_binary)
            ).astype(np.uint8)

        elif self.blend_mode == "mix":
            # Mix with random alpha
            alpha = random.uniform(0.7, 1.0)
            mask_scaled = mask * alpha
            blended = (
                foreground_rgb * mask_scaled + background * (1 - mask_scaled)
            ).astype(np.uint8)

        else:
            # Default to alpha blending
            blended = (foreground_rgb * mask + background * (1 - mask)).astype(np.uint8)

        return blended

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        """Get transform initialization argument names."""
        return (
            "coco_file",
            "object_counts",
            "image_dir",
            "blend_mode",
            "scale_range",
            "min_visibility",
            "rotation_range",
            "include_context",
        )
