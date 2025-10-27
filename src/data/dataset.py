"""
PyTorch Dataset class for flood detection
Handles loading and preprocessing S1+S2 data with flood masks
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from typing import List, Tuple, Optional
import rasterio
from pathlib import Path

from .preprocessing import stack_s1_s2, normalize_bands, quality_control


class FloodDataset(Dataset):
    """
    Dataset for flood detection using Sentinel-1 and Sentinel-2 data
    """
    
    def __init__(
        self,
        s1_paths: List[str],
        s2_paths: List[str],
        label_paths: Optional[List[str]] = None,
        is_train: bool = True,
        augment: bool = True,
        normalize: bool = True,
        tile_size: int = 512
    ):
        """
        Args:
            s1_paths: List of paths to Sentinel-1 images
            s2_paths: List of paths to Sentinel-2 images
            label_paths: List of paths to flood masks (None for inference)
            is_train: Whether this is training data
            augment: Whether to apply data augmentation
            normalize: Whether to normalize bands
            tile_size: Size of image tiles
        """
        self.s1_paths = s1_paths
        self.s2_paths = s2_paths
        self.label_paths = label_paths
        self.is_train = is_train
        self.normalize = normalize
        self.tile_size = tile_size
        
        # Setup data augmentation
        if augment and is_train:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.GaussNoise(p=0.1),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                ToTensorV2()
            ])
    
    def __len__(self) -> int:
        return len(self.s1_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Load and preprocess a single sample
        
        Returns:
            tuple: (image, mask) where mask is None if label_paths is None
        """
        # Load S1 data
        s1_array = self._load_geotiff(self.s1_paths[idx])
        
        # Load S2 data
        s2_array = self._load_geotiff(self.s2_paths[idx])
        
        # Stack S1 and S2 bands
        image = np.concatenate([s1_array, s2_array], axis=0)  # (C, H, W)
        
        # Quality control
        image = quality_control(image)
        
        # Normalize
        if self.normalize:
            image = normalize_bands(image, method='zscore')
        
        # Transpose to (H, W, C) for albumentations
        if len(image.shape) == 3:
            image = np.transpose(image, (1, 2, 0))
        
        # Load mask if available
        mask = None
        if self.label_paths is not None:
            mask = self._load_mask(self.label_paths[idx])
        
        # Apply transformations
        if mask is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask'].long()
        else:
            transformed = self.transform(image=image)
            image = transformed['image']
        
        return image, mask
    
    def _load_geotiff(self, path: str) -> np.ndarray:
        """Load GeoTIFF file"""
        with rasterio.open(path) as src:
            data = src.read()
        return data
    
    def _load_mask(self, path: str) -> np.ndarray:
        """Load binary flood mask"""
        with rasterio.open(path) as src:
            mask = src.read(1)  # Read first band
            
        # Convert to binary if needed
        if mask.max() > 1:
            mask = (mask > 0).astype(np.uint8)
        else:
            mask = mask.astype(np.uint8)
        
        return mask


def create_train_val_test_splits(
    s1_dir: str,
    s2_dir: str,
    labels_dir: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15
) -> Tuple[List[str], List[str], List[str]]:
    """
    Create train/val/test splits
    
    Args:
        s1_dir: Directory containing S1 images
        s2_dir: Directory containing S2 images
        labels_dir: Directory containing labels
        train_ratio: Ratio of training data
        val_ratio: Ratio of validation data
        test_ratio: Ratio of test data
        
    Returns:
        Tuple of (train_paths, val_paths, test_paths)
        Each is a tuple of (s1_paths, s2_paths, label_paths)
    """
    # Get all matching files
    s1_files = sorted(Path(s1_dir).glob("*.tif"))
    s2_files = sorted(Path(s2_dir).glob("*.tif"))
    label_files = sorted(Path(labels_dir).glob("*.tif"))
    
    # Match files by name
    matched = []
    for s1_path in s1_files:
        name = s1_path.stem
        s2_path = Path(s2_dir) / f"{name}.tif"
        label_path = Path(labels_dir) / f"{name}.tif"
        
        if s2_path.exists() and label_path.exists():
            matched.append((str(s1_path), str(s2_path), str(label_path)))
    
    # Shuffle
    np.random.seed(42)
    np.random.shuffle(matched)
    
    # Split
    n_total = len(matched)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    train_files = matched[:n_train]
    val_files = matched[n_train:n_train+n_val]
    test_files = matched[n_train+n_val:]
    
    # Unzip
    train_s1, train_s2, train_labels = zip(*train_files) if train_files else ([], [], [])
    val_s1, val_s2, val_labels = zip(*val_files) if val_files else ([], [], [])
    test_s1, test_s2, test_labels = zip(*test_files) if test_files else ([], [], [])
    
    print(f"Train: {len(train_s1)}, Val: {len(val_s1)}, Test: {len(test_s1)}")
    
    return (
        (list(train_s1), list(train_s2), list(train_labels)),
        (list(val_s1), list(val_s2), list(val_labels)),
        (list(test_s1), list(test_s2), list(test_labels))
    )


if __name__ == "__main__":
    # Example usage
    s1_dir = "data/raw/S1/"
    s2_dir = "data/raw/S2/"
    labels_dir = "data/raw/labels/"
    
    # Create splits
    train_data, val_data, test_data = create_train_val_test_splits(
        s1_dir, s2_dir, labels_dir
    )
    
    # Create datasets
    train_dataset = FloodDataset(
        train_data[0], train_data[1], train_data[2],
        is_train=True, augment=True
    )
    
    val_dataset = FloodDataset(
        val_data[0], val_data[1], val_data[2],
        is_train=False, augment=False
    )
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")

