"""
Cache Dataset for efficient data loading.

Supports partial caching of dataset in memory for faster training.
"""

import logging
from copy import deepcopy
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)


class CacheDataset(Dataset):
    """
    Dataset with optional in-memory caching for faster repeated access.
    
    Args:
        data: List of dicts with 'image' (path) and 'label' keys
        transform: Albumentations transform to apply
        cache_rate: Fraction of data to cache (0.0 to 1.0)
        num_workers: Workers for parallel cache loading
        data_root: Optional root path to prepend to image paths
    """
    
    def __init__(
        self,
        data: List[Dict[str, Any]],
        transform: Optional[Callable] = None,
        cache_rate: float = 1.0,
        num_workers: int = 4,
        data_root: Optional[Path] = None,
    ):
        self.data = data
        self.transform = transform
        self.cache_rate = cache_rate
        self.num_workers = num_workers
        self.data_root = Path(data_root) if data_root else None
        
        self.cache_num = 0
        self._cache: List[Optional[Dict[str, np.ndarray]]] = []
        self._failed_indices: List[int] = []
        
        self._setup_cache()
    
    def _setup_cache(self) -> None:
        """Initialize the cache with parallel loading."""
        length = len(self.data)
        self.cache_num = min(int(length * self.cache_rate), length)
        
        if self.cache_num == 0:
            logger.info("Caching disabled (cache_rate=0)")
            return
        
        indices = list(range(self.cache_num))
        
        with ThreadPool(self.num_workers) as pool:
            self._cache = list(
                tqdm(
                    pool.imap(self._load_cache_item, indices),
                    total=len(indices),
                    desc="Loading dataset into cache",
                )
            )
        
        # Report any failures
        if self._failed_indices:
            logger.warning(
                f"Failed to cache {len(self._failed_indices)} images: "
                f"{self._failed_indices[:10]}{'...' if len(self._failed_indices) > 10 else ''}"
            )
    
    def _resolve_path(self, image_path: str) -> Path:
        """Resolve image path, handling relative/absolute paths."""
        path = Path(image_path)
        
        if path.is_absolute():
            return path
        
        if self.data_root:
            return self.data_root / path
        
        return path
    
    def _load_cache_item(self, idx: int) -> Optional[Dict[str, np.ndarray]]:
        """
        Load a single item for caching.
        
        FIX: Added error handling for corrupt/missing images.
        Returns None for failed loads instead of crashing.
        """
        try:
            image_path = self._resolve_path(self.data[idx]['image'])
            
            # Validate file exists
            if not image_path.exists():
                logger.warning(f"Image not found: {image_path}")
                self._failed_indices.append(idx)
                return None
            
            # Load image
            image = Image.open(image_path)
            
            # Convert to grayscale if needed (chest X-rays are grayscale)
            if image.mode != 'L':
                image = image.convert('L')
            
            image_array = np.array(image)
            
            # Validate image data
            if image_array.size == 0:
                logger.warning(f"Empty image: {image_path}")
                self._failed_indices.append(idx)
                return None
            
            label = self.data[idx]['label']
            
            return {
                "image": np.ascontiguousarray(image_array),
                "label": np.ascontiguousarray(label),
            }
            
        except Exception as e:
            logger.warning(f"Failed to load image at index {idx}: {e}")
            self._failed_indices.append(idx)
            return None
    
    def _load_item(self, idx: int) -> Dict[str, np.ndarray]:
        """Load item directly from disk (non-cached path)."""
        image_path = self._resolve_path(self.data[idx]['image'])
        
        try:
            image = Image.open(image_path)
            if image.mode != 'L':
                image = image.convert('L')
            image_array = np.array(image)
        except Exception as e:
            logger.error(f"Failed to load {image_path}: {e}")
            # Return a placeholder to avoid crashing training
            image_array = np.zeros((224, 224), dtype=np.uint8)
        
        label = self.data[idx]['label']
        
        return {
            "image": image_array,
            "label": np.array(label),
        }
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """
        Get a single item with transforms applied.
        
        Returns:
            Dict with 'image' tensor [1, H, W] and 'label' tensor [num_classes]
        """
        # Check if item is cached
        if index < self.cache_num and self._cache[index] is not None:
            # Use cached data (deepcopy to avoid modifying cache)
            data = deepcopy(self._cache[index])
        else:
            # Load from disk
            data = self._load_item(index)
        
        image = data['image']
        label = data['label']
        
        # Apply transforms
        if self.transform is not None:
            augmented = self.transform(image=image)
            image = augmented["image"]
        
        # Add channel dimension: [H, W] -> [1, H, W]
        image = np.expand_dims(image, 0).astype(np.float32)
        
        return {
            "image": torch.from_numpy(image),
            "label": torch.tensor(label, dtype=torch.float32),
        }
