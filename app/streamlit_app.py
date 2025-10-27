"""
Streamlit web application for flood detection
"""

import streamlit as st
import numpy as np
import torch
from pathlib import Path
import sys
from typing import Optional
import tempfile
import os

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.inference.predict import FloodPredictor
from src.data.gee_downloader import GEEDownloader
from src.data.preprocessing import stack_s1_s2, normalize_bands
from app.utils import (
    overlay_mask_on_image,
    visualize_uncertainty,
    calculate_flood_area,
    generate_statistics
)


# Page configuration
st.set_page_config(
    page_title="Flood Detection System",
    page_icon="🌊",
    layout="wide"
)

# Title
st.title("🌊 Flood Detection System")
st.markdown("Automated flood detection using Sentinel-1 SAR + Sentinel-2 optical data")

# Sidebar
st.sidebar.header("Configuration")

# Model selection
model_paths_default = {
    'resnet': 'models/resnet_best.pt',
    'swin': 'models/swin_best.pt',
    'maxvit': 'models/maxvit_best.pt'
}

# Check if models exist
available_models = []
for name, path in model_paths_default.items():
    if os.path.exists(path):
        available_models.append(name)

if not available_models:
    st.error("No trained models found. Please train models first.")
    st.stop()

model_selection = st.sidebar.multiselect(
    "Select Models",
    options=available_models,
    default=available_models
)

# Input mode
input_mode = st.sidebar.radio(
    "Input Mode",
    options=["Upload Files", "Google Earth Engine"]
)

# Main content
if input_mode == "Upload Files":
    st.header("Upload Satellite Images")
    
    col1, col2 = st.columns(2)
    
    with col1:
        s1_file = st.file_uploader(
            "Upload Sentinel-1 SAR Image (GeoTIFF)",
            type=['tif', 'tiff'],
            help="Sentinel-1 SAR image with VV and VH polarization bands"
        )
    
    with col2:
        s2_file = st.file_uploader(
            "Upload Sentinel-2 Optical Image (GeoTIFF)",
            type=['tif', 'tiff'],
            help="Sentinel-2 optical image with RGB, NIR, SWIR bands"
        )
    
    if s1_file and s2_file:
        # Save uploaded files
        with tempfile.NamedTemporaryFile(delete=False, suffix='.tif') as tmp_s1:
            tmp_s1.write(s1_file.read())
            s1_path = tmp_s1.name
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.tif') as tmp_s2:
            tmp_s2.write(s2_file.read())
            s2_path = tmp_s2.name
        
        # Process button
        if st.button("🌊 Detect Flood", type="primary"):
            with st.spinner("Processing..."):
                # Load model
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model_paths = [model_paths_default[m] for m in model_selection]
                
                try:
                    predictor = FloodPredictor(model_paths, device=device)
                    
                    # Predict
                    results = predictor.predict_with_uncertainty(s1_path, s2_path)
                    
                    pred = results['prediction']
                    probs = results['probabilities']
                    uncertainty = results['uncertainty']
                    
                    # Display results
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.subheader("Original Image")
                        # Load S2 as RGB
                        import rasterio
                        with rasterio.open(s2_path) as src:
                            s2_rgb = src.read([3, 2, 1])
                            s2_rgb = np.transpose(s2_rgb, (1, 2, 0))
                            s2_rgb = (s2_rgb - s2_rgb.min()) / (s2_rgb.max() - s2_rgb.min())
                        
                        st.image(s2_rgb, use_container_width=True)
                    
                    with col2:
                        st.subheader("Flood Prediction")
                        overlay = overlay_mask_on_image(s2_rgb, pred, alpha=0.5)
                        st.image(overlay, use_container_width=True)
                    
                    with col3:
                        st.subheader("Uncertainty Map")
                        uncertainty_vis = visualize_uncertainty(uncertainty)
                        st.image(uncertainty_vis, use_container_width=True)
                    
                    # Statistics
                    st.subheader("📊 Statistics")
                    stats = generate_statistics(pred, uncertainty)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Flood Percentage", f"{stats['flood_percentage']:.2f}%")
                    with col2:
                        st.metric("Flood Pixels", f"{stats['flood_pixels']:,}")
                    with col3:
                        st.metric("Mean Uncertainty", f"{stats.get('mean_uncertainty', 0):.3f}")
                    with col4:
                        st.metric("High Uncertainty", f"{stats.get('high_uncertainty_pixels', 0):,}")
                    
                    # Download options
                    st.subheader("📥 Download Results")
                    # TODO: Implement download functionality
                    st.info("Download functionality coming soon")
                    
                except Exception as e:
                    st.error(f"Error during prediction: {str(e)}")
                    st.exception(e)
        
        # Cleanup
        os.unlink(s1_path)
        os.unlink(s2_path)


elif input_mode == "Google Earth Engine":
    st.header("Download from Google Earth Engine")
    
    st.info("This feature requires Google Earth Engine authentication")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Region of Interest")
        min_lon = st.number_input("Minimum Longitude", value=77.2, step=0.1)
        min_lat = st.number_input("Minimum Latitude", value=28.4, step=0.1)
    
    with col2:
        st.subheader("Date Range")
        start_date = st.date_input("Start Date", value=None)
        end_date = st.date_input("End Date", value=None)
    
    if st.button("Download & Process"):
        with st.spinner("Downloading from Google Earth Engine..."):
            try:
                # Initialize GEE
                downloader = GEEDownloader()
                
                # Download data
                s1_path, s2_path = downloader.download_co_registered(
                    coords=(min_lon, min_lat, min_lon + 0.1, min_lat + 0.1),
                    start_date=start_date.isoformat(),
                    end_date=end_date.isoformat(),
                    output_dir="data/raw/",
                    prefix="gee_flood"
                )
                
                st.success("Data downloaded successfully!")
                st.info(f"S1: {s1_path}")
                st.info(f"S2: {s2_path}")
                
            except Exception as e:
                st.error(f"Error downloading data: {str(e)}")
                st.exception(e)

# Footer
st.markdown("---")
st.markdown("""
### About
This flood detection system uses deep learning ensemble models trained on Sentinel-1 and Sentinel-2 satellite data.

**Features:**
- Multi-model ensemble (ResNet-50, Swin Transformer, MaxViT)
- Uncertainty quantification
- Automated data fetching from Google Earth Engine
- Interactive visualization

**Citation:**
Sharma, N.K., Saharia, M., 2025. DeepSARFlood: Rapid and Automated SAR-based flood inundation mapping using Vision Transformer-based Deep Ensembles with uncertainty estimates.
""")


if __name__ == "__main__":
    # Run with: streamlit run app/streamlit_app.py
    pass

