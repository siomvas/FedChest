"""
Common image transformation functions for FedChest preprocessing.

All datasets use the same transforms to ensure consistency.

FIXES from original notebooks:
- CXR14 used CLAHE clipLimit=2.0
- CheXpert used CLAHE clipLimit=1.0  <-- INCONSISTENT
- PadChest used CLAHE clipLimit=2.0
Now standardized to 2.0 for all datasets.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def load_image(
    image_path: Union[str, Path],
    grayscale: bool = True,
) -> Optional[np.ndarray]:
    """
    Load an image from disk.
    
    Args:
        image_path: Path to the image file
        grayscale: If True, load as grayscale
    
    Returns:
        Image as numpy array, or None if loading fails
    """
    path = Path(image_path)
    
    if not path.exists():
        logger.warning(f"Image not found: {path}")
        return None
    
    try:
        flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
        img = cv2.imread(str(path), flag)
        
        if img is None:
            logger.warning(f"Failed to decode image: {path}")
            return None
        
        return img
    
    except Exception as e:
        logger.warning(f"Error loading {path}: {e}")
        return None


def crop_borders(
    image: np.ndarray,
    crop_percent: float = 0.05,
) -> np.ndarray:
    """
    Crop a percentage from each edge to remove borders and artifacts.
    
    Chest X-rays often have blown-out borders, patient information
    overlays, or other artifacts near the edges.
    
    Args:
        image: Input image array
        crop_percent: Fraction to crop from each edge (default 5%)
    
    Returns:
        Cropped image
    """
    h, w = image.shape[:2]
    
    top = int(h * crop_percent)
    bottom = int(h * (1 - crop_percent))
    left = int(w * crop_percent)
    right = int(w * (1 - crop_percent))
    
    return image[top:bottom, left:right]


def resize_image(
    image: np.ndarray,
    size: Tuple[int, int] = (224, 224),
    interpolation: int = cv2.INTER_LINEAR,
) -> np.ndarray:
    """
    Resize image to target size.
    
    Args:
        image: Input image array
        size: Target (width, height)
        interpolation: OpenCV interpolation method
    
    Returns:
        Resized image
    """
    return cv2.resize(image, size, interpolation=interpolation)


def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8),
) -> np.ndarray:
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).
    
    CLAHE improves contrast in medical images while avoiding 
    over-amplification of noise.
    
    CRITICAL: clip_limit is standardized to 2.0 for ALL datasets.
    Original notebooks had inconsistent values (CXP used 1.0).
    
    Args:
        image: Input grayscale image
        clip_limit: Threshold for contrast limiting (standardized to 2.0)
        tile_grid_size: Size of grid for histogram equalization
    
    Returns:
        Contrast-enhanced image
    """
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid_size,
    )
    return clahe.apply(image)


def preprocess_chest_xray(
    image_path: Union[str, Path],
    output_path: Union[str, Path],
    size: Tuple[int, int] = (224, 224),
    crop_percent: float = 0.05,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8),
) -> bool:
    """
    Full preprocessing pipeline for a single chest X-ray image.
    
    Pipeline:
    1. Load grayscale image
    2. Crop borders (5% from each edge)
    3. Resize to target size (224x224)
    4. Apply CLAHE for contrast enhancement
    5. Save processed image
    
    Args:
        image_path: Path to source image
        output_path: Path to save processed image
        size: Target image size
        crop_percent: Border crop percentage
        clip_limit: CLAHE clip limit (standardized to 2.0)
        tile_grid_size: CLAHE tile grid size
    
    Returns:
        True if successful, False otherwise
    """
    # Load
    image = load_image(image_path, grayscale=True)
    if image is None:
        return False
    
    # Crop borders
    image = crop_borders(image, crop_percent)
    
    # Resize
    image = resize_image(image, size)
    
    # CLAHE
    image = apply_clahe(image, clip_limit, tile_grid_size)
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        cv2.imwrite(str(output_path), image)
        return True
    except Exception as e:
        logger.warning(f"Failed to save {output_path}: {e}")
        return False


def verify_image(image_path: Union[str, Path]) -> bool:
    """
    Verify that an image file is valid and loadable.
    
    Args:
        image_path: Path to image file
    
    Returns:
        True if image is valid, False otherwise
    """
    img = load_image(image_path)
    return img is not None and img.size > 0
