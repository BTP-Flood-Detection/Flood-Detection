"""
Simplified download script - provides data download guidance
"""

import os
from pathlib import Path

def main():
    """Main function"""
    
    print("="*60)
    print("FLOOD DETECTION - DATASET DOWNLOADER")
    print("="*60)
    print()
    print("OPTIONS FOR GETTING TRAINING DATA:")
    print()
    print("Option 1: Use Sen1Floods11 Dataset (Recommended)")
    print("   1. Visit: https://github.com/cloudtostreet/Sen1Floods11")
    print("   2. Download ~10-15 sample images (small dataset)")
    print("   3. Place in:")
    print("      - data/raw/S1/   <- Sentinel-1 SAR images")
    print("      - data/raw/S2/   <- Sentinel-2 optical images")
    print("      - data/raw/labels/ <- Flood mask labels")
    print()
    print("Option 2: Use the Streamlit App (For Testing)")
    print("   1. Run: streamlit run app/streamlit_app.py")
    print("   2. Use 'Google Earth Engine' mode")
    print("   3. Specify coordinates and download fresh data")
    print()
    print("Option 3: Google Colab for Easy Download")
    print("   Create a Colab notebook and use Earth Engine there")
    print("   Much easier than Windows GDAL setup")
    print()
    print("="*60)
    print("CURRENT PROJECT STATUS:")
    print("="*60)
    print("All code files implemented")
    print("Models: ResNet, Swin, MaxViT ready")
    print("Training script ready")
    print("Streamlit app ready")
    print()
    print("NEEDED FOR TRAINING:")
    print("- Download ~1-2 GB of Sen1Floods11 data")
    print("- Run: python src/training/train.py --model resnet")
    print("="*60)


if __name__ == "__main__":
    main()
