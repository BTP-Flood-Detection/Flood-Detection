"""
Quick setup script for the flood detection project
"""

import subprocess
import sys
from pathlib import Path

def install_requirements():
    """Install required packages"""
    print("📦 Installing requirements...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    print("✅ Requirements installed\n")

def setup_gee():
    """Setup Google Earth Engine"""
    print("🌍 Setting up Google Earth Engine...")
    print("Running: earthengine authenticate")
    try:
        subprocess.run(["earthengine", "authenticate"], check=True)
        print("✅ Google Earth Engine authenticated\n")
    except subprocess.CalledProcessError:
        print("⚠️  Please run: earthengine authenticate")
        print("Visit: https://earthengine.google.com/")
        print()

def create_structure():
    """Create necessary directories"""
    print("📁 Creating project structure...")
    dirs = [
        "data/raw/S1",
        "data/raw/S2", 
        "data/raw/labels",
        "data/processed",
        "models",
        "logs",
        "notebooks"
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    print("✅ Directory structure created\n")

def main():
    print("="*60)
    print("FLOOD DETECTION SYSTEM - SETUP")
    print("="*60)
    print()
    
    # Create structure
    create_structure()
    
    # Install requirements
    print("Would you like to install requirements now? (y/n): ", end="")
    choice = input().strip().lower()
    if choice == 'y':
        install_requirements()
    
    # Setup GEE
    print("Would you like to setup Google Earth Engine? (y/n): ", end="")
    choice = input().strip().lower()
    if choice == 'y':
        setup_gee()
    
    print("="*60)
    print("✅ Setup complete!")
    print("\nNext steps:")
    print("1. Download sample data: python download_sample_data.py")
    print("2. Train models: python src/training/train.py --model resnet")
    print("3. Run web app: streamlit run app/streamlit_app.py")
    print("="*60)

if __name__ == "__main__":
    main()

