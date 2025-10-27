"""
Utility functions for Streamlit application
Image visualization, conversion, and report generation
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import rasterio
from pathlib import Path
from typing import Tuple, Optional


def load_geotiff_as_rgb(path: str, bands: list = [3, 2, 1]) -> np.ndarray:
    """
    Load GeoTIFF and convert to RGB for display
    
    Args:
        path: Path to GeoTIFF file
        bands: List of band indices for RGB (default: [3, 2, 1] for RGB)
        
    Returns:
        RGB image array (H, W, 3)
    """
    with rasterio.open(path) as src:
        data = src.read()
        
    # Select bands and normalize
    rgb = np.stack([data[i-1] for i in bands], axis=-1)
    
    # Normalize to 0-1 range
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-10)
    rgb = np.clip(rgb, 0, 1)
    
    return rgb


def overlay_mask_on_image(
    image: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.5,
    color: Tuple[int, int, int] = (255, 0, 0)
) -> np.ndarray:
    """
    Overlay flood mask on image
    
    Args:
        image: Base image (H, W, 3)
        mask: Binary flood mask (H, W)
        alpha: Transparency of overlay
        color: Color for flood pixels (RGB)
        
    Returns:
        Overlayed image
    """
    overlay = image.copy()
    
    # Create colored mask
    colored_mask = np.zeros_like(image)
    colored_mask[mask == 1] = color
    
    # Blend
    mask_binary = (mask == 1)[:, :, np.newaxis]
    overlay[mask_binary] = (
        (1 - alpha) * image[mask_binary] + alpha * colored_mask[mask_binary]
    )
    
    return overlay.astype(np.uint8)


def create_uncertainty_colormap() -> plt.cm:
    """Create colormap for uncertainty visualization"""
    colors = ['green', 'yellow', 'orange', 'red']
    n_bins = 256
    cmap = plt.get_cmap('RdYlGn_r')
    return cmap


def visualize_uncertainty(uncertainty: np.ndarray, cmap: str = 'RdYlGn_r') -> np.ndarray:
    """
    Visualize uncertainty map as colored heatmap
    
    Args:
        uncertainty: Uncertainty map (H, W)
        cmap: Colormap name
        
    Returns:
        Colored uncertainty map (H, W, 3)
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(uncertainty, cmap=cmap, vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label='Uncertainty')
    plt.tight_layout()
    
    # Convert to numpy array
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    
    return img


def calculate_flood_area(mask: np.ndarray, pixel_size: float) -> float:
    """
    Calculate flood area in km²
    
    Args:
        mask: Binary flood mask
        pixel_size: Pixel size in meters
        
    Returns:
        Area in km²
    """
    flood_pixels = np.sum(mask == 1)
    area_m2 = flood_pixels * (pixel_size ** 2)
    area_km2 = area_m2 / 1e6
    
    return area_km2


def generate_statistics(mask: np.ndarray, uncertainty: Optional[np.ndarray] = None) -> dict:
    """
    Generate statistics from flood prediction
    
    Args:
        mask: Binary flood mask
        uncertainty: Uncertainty map (optional)
        
    Returns:
        Dictionary of statistics
    """
    stats = {
        'total_pixels': mask.size,
        'flood_pixels': np.sum(mask == 1),
        'flood_percentage': 100 * np.mean(mask == 1),
        'background_pixels': np.sum(mask == 0),
        'background_percentage': 100 * np.mean(mask == 0)
    }
    
    if uncertainty is not None:
        stats['mean_uncertainty'] = np.mean(uncertainty)
        stats['high_uncertainty_pixels'] = np.sum(uncertainty > 0.7)
    
    return stats


if __name__ == "__main__":
    # Example usage
    image = np.random.rand(512, 512, 3)
    mask = np.random.randint(0, 2, size=(512, 512))
    
    # Overlay mask
    overlay = overlay_mask_on_image(image, mask)
    print(f"Overlay shape: {overlay.shape}")
    
    # Statistics
    stats = generate_statistics(mask)
    for key, value in stats.items():
        print(f"{key}: {value:.4f}")

