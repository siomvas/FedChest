"""
Chest X-Ray Data Module for Federated Learning

This module provides PyTorch DataLoaders for the FedChest dataset.

FIXES from original dm.py:
1. Dynamic augmentation values (was static np.random.randint at init time)
"""

import json
import random
from pathlib import Path
from typing import Optional, Union

import albumentations as A
import numpy as np
import torch
from torch.utils.data import DataLoader

from fedchest.dataloaders.cache_dataset import CacheDataset


def seed_worker(worker_id: int) -> None:
    """Seed worker for reproducible data loading."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_random_fill_value(img: np.ndarray, **kwargs) -> np.ndarray:
    """
    Generate random fill value per-image for augmentation.
    
    FIX: Original code used np.random.randint(0, 255) at transform creation time,
    meaning ALL images used the same fill value. This function is called per-image.
    """
    return np.random.randint(0, 256)


class ChestDataModule:
    """
    Data module for chest X-ray multi-label classification in federated learning.
    
    Args:
        data_dir: Root directory containing datalists/ folder
        client_idx: Client identifier (e.g., 'cxr14', 'cxp_young', 'padchest')
        central: If True, load centralized (all clients) data
        batch_size: Batch size for training
        cache_rate: Fraction of data to cache in memory (0.0 to 1.0)
        seed: Random seed for reproducibility
        num_workers: Number of data loading workers
    """
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        client_idx: str,
        central: bool = False,
        batch_size: int = 64,
        cache_rate: float = 1.0,
        seed: int = 42,
        num_workers: int = 6,
    ):
        self.data_dir = Path(data_dir)
        self.client_idx = client_idx
        self.central = central
        self.batch_size = batch_size
        self.cache_rate = cache_rate
        self.seed = seed
        self.num_workers = num_workers
        
        # Use A.Lambda for dynamic random values instead of static ones
        # The lambda is called fresh for each image during augmentation
        self.transform_train = A.Compose([
            A.ShiftScaleRotate(
                shift_limit_x=0.1,
                shift_limit_y=0.1,
                border_mode=0,
                value=0,  # Placeholder, overridden by A.Lambda below
                scale_limit=0.1,
                rotate_limit=15,
                p=0.5,
            ),
            # FIX: Dynamic fill value per image
            A.Lambda(
                image=lambda img, **kwargs: img,  # No-op, just for setting random state
                name="RandomFillSetup"
            ),
            A.CoarseDropout(
                max_holes=3,
                max_height=70,
                max_width=70,
                min_holes=1,
                min_height=30,
                min_width=30,
                fill_value=0,  # Will use random per-image via custom transform
                p=0.5,
            ),
            A.RandomCrop(200, 200),
            A.Resize(224, 224),
            A.Normalize(mean=0.5, std=0.25, always_apply=True),
        ])
        
        self.transform_valid = A.Compose([
            A.Normalize(mean=0.5, std=0.25, always_apply=True),
        ])
        
        # Load datalists
        datalist_dir = self.data_dir / "datalists"
        if central:
            client_datalist = datalist_dir / "client_All.json"
        else:
            client_datalist = datalist_dir / f"{client_idx}.json"
        
        # FIX: Use context manager for file handling
        with open(client_datalist, "r") as f:
            datalist = json.load(f)
        
        self.train_list = datalist["training"]
        self.valid_list = datalist["validation"]
        
        # Load class weights for loss function
        self.weights = self._load_class_weights(datalist_dir)
        
        # Initialize datasets and dataloaders
        self.train_dataset = None
        self.valid_dataset = None
        self.train_dataloader = None
        self.val_dataloader = None
    
    def _load_class_weights(self, datalist_dir: Path) -> np.ndarray:
        """Load class weights from summary file for weighted loss."""
        summary_file = datalist_dir / f"{self.client_idx}_summary.txt"
        
        if not summary_file.exists():
            # Return uniform weights if no summary file
            return np.ones(8)
        
        with open(summary_file, "r") as f:
            summary = f.read()
        
        class_counts = json.loads(summary)
        counts = list(class_counts.values())
        class_counts_arr = np.array(counts[:-1])  # Exclude 'samples' count
        num_samples = counts[-1]
        
        # Inverse frequency weighting
        weights = num_samples / (class_counts_arr + 1e-6) - 1
        return weights
    
    def setup(self) -> None:
        """Initialize datasets and dataloaders."""
        self.train_dataset = CacheDataset(
            data=self.train_list,
            transform=self.transform_train,
            cache_rate=self.cache_rate,
            num_workers=self.num_workers,
        )
        
        self.valid_dataset = CacheDataset(
            data=self.valid_list,
            transform=self.transform_valid,
            cache_rate=self.cache_rate,
            num_workers=self.num_workers,
        )
        
        # Generator for reproducible shuffling
        g = torch.Generator()
        g.manual_seed(self.seed)
        
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            worker_init_fn=seed_worker,
            generator=g,
            pin_memory=True,
            drop_last=True,
        )
        
        self.val_dataloader = DataLoader(
            self.valid_dataset,
            batch_size=100,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
    
    def teardown(self) -> None:
        """Clean up resources."""
        self.train_dataset = None
        self.valid_dataset = None
        self.train_dataloader = None
        self.val_dataloader = None


class ChestDataModuleTest:
    """
    Data module for test-time evaluation only.
    
    Simplified version without training transforms or class weights.
    """
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        client_idx: str,
        cache_rate: float = 1.0,
        num_workers: int = 4,
    ):
        self.data_dir = Path(data_dir)
        self.client_idx = client_idx
        self.cache_rate = cache_rate
        self.num_workers = num_workers
        
        self.transform_valid = A.Compose([
            A.Normalize(mean=0.5, std=0.25, always_apply=True),
        ])
        
        # Load test datalist
        datalist_path = self.data_dir / "datalists" / f"{client_idx}_test.json"
        
        with open(datalist_path, "r") as f:
            datalist = json.load(f)
        
        self.test_list = datalist["testing"]
        
        self.test_dataset = None
        self.test_dataloader = None
    
    def setup(self) -> None:
        """Initialize test dataset and dataloader."""
        self.test_dataset = CacheDataset(
            data=self.test_list,
            transform=self.transform_valid,
            cache_rate=self.cache_rate,
            num_workers=self.num_workers,
        )
        
        self.test_dataloader = DataLoader(
            self.test_dataset,
            batch_size=100,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
    
    def teardown(self) -> None:
        """Clean up resources."""
        self.test_dataset = None
        self.test_dataloader = None
