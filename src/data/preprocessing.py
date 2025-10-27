"""
Data preprocessing utilities for flood detection
Handles alignment, normalization, and quality control
"""

import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.enums import Resampling as ResamplingEnum
import cv2
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


def align_images(
    s1_path: str,
    s2_path: str,
    output_path: str,
    target_resolution: float = 10.0,
    reference_crs: str = 'EPSG:4326'
) -> Tuple[str, str]:
    """
    Align S2 to S1 spatial resolution and extent
    
    Args:
        s1_path: Path to Sentinel-1 GeoTIFF
        s2_path: Path to Sentinel-2 GeoTIFF
        output_path: Path to save aligned S2
        target_resolution: Target spatial resolution in meters
        reference_crs: Target CRS
        
    Returns:
        Tuple of (aligned_s1_path, aligned_s2_path)
    """
    with rasterio.open(s1_path) as s1_src:
        # Get S1 metadata
        s1_profile = s1_src.profile
        s1_bounds = s1_src.bounds
        s1_crs = s1_src.crs
        
        # Reproject S2 to match S1
        with rasterio.open(s2_path) as s2_src:
            # Calculate transform
            transform, width, height = calculate_default_transform(
                s2_src.crs,
                s1_crs,
                s1_src.width,
                s1_src.height,
                *s1_bounds,
                resolution=target_resolution
            )
            
            s2_profile = s2_src.profile.copy()
            s2_profile.update({
                'crs': s1_crs,
                'transform': transform,
                'width': width,
                'height': height
            })
            
            # Reproject S2 to S1's CRS and bounds
            with rasterio.open(output_path, 'w', **s2_profile) as dst:
                for i in range(1, s2_src.count + 1):
                    reproject(
                        source=rasterio.band(s2_src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=s2_src.transform,
                        src_crs=s2_src.crs,
                        dst_transform=transform,
                        dst_crs=s1_crs,
                        resampling=ResamplingEnum.bilinear
                    )
    
    print(f"Aligned S2 saved to: {output_path}")
    return s1_path, output_path


def normalize_bands(
    image: np.ndarray,
    method: str = 'zscore'
) -> np.ndarray:
    """
    Normalize image bands
    
    Args:
        image: Image array (H, W, C) or (C, H, W)
        method: Normalization method ('zscore', 'minmax', 'percentile')
        
    Returns:
        Normalized image
    """
    normalized = image.copy().astype(np.float32)
    
    # Handle different input shapes
    if len(normalized.shape) == 3 and normalized.shape[0] < 100:  # (C, H, W)
        # Channel-first format
        for i in range(normalized.shape[0]):
            band = normalized[i]
            
            if method == 'zscore':
                mean = np.nanmean(band)
                std = np.nanstd(band)
                if std > 0:
                    normalized[i] = (band - mean) / std
            elif method == 'minmax':
                min_val = np.nanmin(band)
                max_val = np.nanmax(band)
                if max_val > min_val:
                    normalized[i] = (band - min_val) / (max_val - min_val)
            elif method == 'percentile':
                p2 = np.nanpercentile(band, 2)
                p98 = np.nanpercentile(band, 98)
                if p98 > p2:
                    normalized[i] = np.clip((band - p2) / (p98 - p2), 0, 1)
    
    return normalized


def stack_s1_s2(
    s1_path: str,
    s2_path: str,
    s1_bands: int = 2,
    s2_bands: int = 4
) -> np.ndarray:
    """
    Stack Sentinel-1 and Sentinel-2 bands into single array
    
    Args:
        s1_path: Path to S1 GeoTIFF
        s2_path: Path to S2 GeoTIFF
        s1_bands: Number of S1 bands (default: 2 for VV, VH)
        s2_bands: Number of S2 bands to use (default: 4 for RGB+NIR)
        
    Returns:
        Stacked array (C, H, W) where C = s1_bands + s2_bands
    """
    with rasterio.open(s1_path) as s1_src:
        s1_data = s1_src.read()  # (C, H, W)
        
    with rasterio.open(s2_path) as s2_src:
        s2_data = s2_src.read()[:s2_bands]  # Use first s2_bands bands
    
    # Stack bands
    stacked = np.concatenate([s1_data, s2_data], axis=0)
    
    return stacked


def create_tiles(
    image: np.ndarray,
    tile_size: int = 512,
    overlap: int = 64
) -> list:
    """
    Split large image into overlapping tiles
    
    Args:
        image: Image array (C, H, W)
        tile_size: Size of each tile
        overlap: Overlap between tiles in pixels
        
    Returns:
        List of tile arrays
    """
    if len(image.shape) == 3:
        _, h, w = image.shape
    else:
        h, w = image.shape[:2]
    
    tiles = []
    stride = tile_size - overlap
    
    for i in range(0, h - tile_size + 1, stride):
        for j in range(0, w - tile_size + 1, stride):
            if len(image.shape) == 3:
                tile = image[:, i:i+tile_size, j:j+tile_size]
            else:
                tile = image[i:i+tile_size, j:j+tile_size]
            tiles.append(tile)
    
    return tiles


def quality_control(
    image: np.ndarray,
    no_data_value: float = -9999
) -> np.ndarray:
    """
    Remove invalid pixels and handle NaN values
    
    Args:
        image: Image array
        no_data_value: No-data value to replace
        
    Returns:
        Cleaned image
    """
    cleaned = image.copy()
    
    # Handle NaN and Inf
    cleaned = np.nan_to_num(cleaned, nan=no_data_value, posinf=no_data_value, neginf=no_data_value)
    
    # Remove extreme outliers
    q1 = np.percentile(cleaned, 0.01)
    q99 = np.percentile(cleaned, 99.99)
    cleaned = np.clip(cleaned, q1, q99)
    
    return cleaned


def apply_morphological_operations(
    mask: np.ndarray,
    operation: str = 'closing',
    kernel_size: int = 3
) -> np.ndarray:
    """
    Apply morphological operations to clean flood mask
    
    Args:
        mask: Binary flood mask
        operation: Type of operation ('opening', 'closing', 'both')
        kernel_size: Size of structuring element
        
    Returns:
        Cleaned mask
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    if operation == 'opening':
        cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    elif operation == 'closing':
        cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    elif operation == 'both':
        cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    else:
        cleaned = mask
    
    return cleaned


if __name__ == "__main__":
    # Example usage
    s1_path = "data/raw/S1/test.tif"
    s2_path = "data/raw/S2/test.tif"
    
    # Align images
    aligned_s1, aligned_s2 = align_images(s1_path, s2_path, "data/processed/S2_aligned.tif")
    
    # Stack bands
    stacked = stack_s1_s2(aligned_s1, aligned_s2)
    
    # Normalize
    normalized = normalize_bands(stacked)
    
    print(f"Stacked array shape: {stacked.shape}")
    print(f"Normalized array shape: {normalized.shape}")

