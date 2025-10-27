"""
Google Earth Engine Data Downloader for Sentinel-1 and Sentinel-2
Automates fetching SAR and optical imagery for flood detection
"""

import os
import ee
import geemap
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from typing import List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class GEEDownloader:
    """Download Sentinel-1 and Sentinel-2 data from Google Earth Engine"""
    
    def __init__(self):
        """Initialize Earth Engine"""
        try:
            ee.Initialize()
            print("Earth Engine initialized successfully")
        except Exception as e:
            print(f"Earth Engine authentication needed. Run: earthengine authenticate")
            raise
    
    def download_sentinel1(
        self,
        aoi: ee.Geometry,
        start_date: str,
        end_date: str,
        output_path: str,
        export_bands: List[str] = ['VV', 'VH']
    ) -> str:
        """
        Download Sentinel-1 SAR data
        
        Args:
            aoi: Area of Interest as Earth Engine Geometry
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            output_path: Path to save the GeoTIFF
            export_bands: List of polarization bands to download
            
        Returns:
            Path to downloaded file
        """
        # Load Sentinel-1 collection
        s1_collection = (ee.ImageCollection('COPERNICUS/S1_GRD')
                        .filterBounds(aoi)
                        .filterDate(start_date, end_date)
                        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')))
        
        # Select first available image
        s1_image = s1_collection.first()
        
        if s1_image is None:
            raise ValueError("No Sentinel-1 images found for the specified dates and location")
        
        # Select bands
        s1_image = s1_image.select(export_bands)
        
        # Apply speckle filtering (Lee filter)
        def apply_lee_filter(img):
            vv = img.select('VV')
            vh = img.select('VH')
            
            # Lee filter for VV
            vv_filtered = ee.Image().convolve(
                ee.Kernel.square(radius=3, units='pixels')
            ).reduceNeighborhood(
                reducer=ee.Reducer.stdDev(),
                kernel=ee.Kernel.square(radius=3, units='pixels')
            )
            
            return img.addBands(vv_filtered, None, True)
        
        # Get image bounds and CRS
        bounds = s1_image.geometry().bounds().getInfo()
        transform = from_bounds(
            bounds['coordinates'][0][0][0],  # left
            bounds['coordinates'][0][0][1],  # bottom
            bounds['coordinates'][0][2][0],  # right
            bounds['coordinates'][0][2][1],  # top
            256, 256
        )
        
        # Download as numpy array
        s1_array = geemap.ee_to_numpy(s1_image, aoi=aoi, scale=10)
        
        # Convert to GeoTIFF
        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=s1_array.shape[0],
            width=s1_array.shape[1],
            count=len(export_bands),
            dtype=rasterio.float32,
            crs='EPSG:4326',
            transform=transform
        ) as dst:
            for i, band in enumerate(export_bands):
                dst.write(s1_array[:, :, i] if len(s1_array.shape) == 3 else s1_array, i + 1)
        
        print(f"Sentinel-1 data saved to {output_path}")
        return output_path
    
    def download_sentinel2(
        self,
        aoi: ee.Geometry,
        start_date: str,
        end_date: str,
        output_path: str,
        cloud_threshold: float = 20.0
    ) -> str:
        """
        Download Sentinel-2 optical data
        
        Args:
            aoi: Area of Interest as Earth Engine Geometry
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            output_path: Path to save the GeoTIFF
            cloud_threshold: Maximum cloud cover percentage
            
        Returns:
            Path to downloaded file
        """
        # Load Sentinel-2 collection
        s2_collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                        .filterBounds(aoi)
                        .filterDate(start_date, end_date)
                        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_threshold)))
        
        # Select first available image
        s2_image = s2_collection.first()
        
        if s2_image is None:
            raise ValueError("No Sentinel-2 images found for the specified dates and location")
        
        # Select bands: Red, Green, Blue, NIR, SWIR1, SWIR2
        s2_bands = ['B4', 'B3', 'B2', 'B8', 'B11', 'B12']  # R, G, B, NIR, SWIR1, SWIR2
        s2_image = s2_image.select(s2_bands)
        
        # Get image bounds and CRS
        bounds = s2_image.geometry().bounds().getInfo()
        transform = from_bounds(
            bounds['coordinates'][0][0][0],  # left
            bounds['coordinates'][0][0][1],  # bottom
            bounds['coordinates'][0][2][0],  # right
            bounds['coordinates'][0][2][1],  # top
            256, 256
        )
        
        # Download as numpy array
        s2_array = geemap.ee_to_numpy(s2_image, aoi=aoi, scale=10)
        
        # Convert to GeoTIFF
        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=s2_array.shape[0],
            width=s2_array.shape[1],
            count=len(s2_bands),
            dtype=rasterio.float32,
            crs='EPSG:4326',
            transform=transform
        ) as dst:
            for i in range(len(s2_bands)):
                dst.write(s2_array[:, :, i] if len(s2_array.shape) == 3 else s2_array, i + 1)
        
        print(f"Sentinel-2 data saved to {output_path}")
        return output_path
    
    def download_co_registered(
        self,
        coords: Tuple[float, float, float, float],  # min_lon, min_lat, max_lon, max_lat
        start_date: str,
        end_date: str,
        output_dir: str,
        prefix: str = "flood_detection"
    ) -> Tuple[str, str]:
        """
        Download spatially aligned S1 and S2 images
        
        Args:
            coords: Bounding box coordinates (min_lon, min_lat, max_lon, max_lat)
            start_date: Start date
            end_date: End date
            output_dir: Directory to save files
            prefix: Prefix for output files
            
        Returns:
            Tuple of (S1_path, S2_path)
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Create AOI geometry
        aoi = ee.Geometry.Rectangle(
            coords,
            proj='EPSG:4326',
            geodesic=False
        )
        
        # Download S1
        s1_path = os.path.join(output_dir, f"{prefix}_S1.tif")
        self.download_sentinel1(aoi, start_date, end_date, s1_path)
        
        # Download S2
        s2_path = os.path.join(output_dir, f"{prefix}_S2.tif")
        self.download_sentinel2(aoi, start_date, end_date, s2_path)
        
        return s1_path, s2_path


if __name__ == "__main__":
    # Example usage
    downloader = GEEDownloader()
    
    # Example: Download data for a flood event
    # Bounding box for example region (adjust coordinates as needed)
    coords = (77.2, 28.4, 77.3, 28.5)  # Example: Delhi region
    start_date = "2023-07-15"
    end_date = "2023-07-20"
    
    s1_path, s2_path = downloader.download_co_registered(
        coords=coords,
        start_date=start_date,
        end_date=end_date,
        output_dir="data/raw/",
        prefix="test_flood"
    )
    
    print(f"\nDownloaded S1 to: {s1_path}")
    print(f"Downloaded S2 to: {s2_path}")

