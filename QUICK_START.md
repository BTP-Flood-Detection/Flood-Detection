# Quick Start Guide

## Current Status

✅ **All code is implemented!**
- Model architectures (ResNet, Swin, MaxViT)
- Training pipeline
- Streamlit web application
- Google Earth Engine integration

## Next Steps - Choose One:

### Option 1: Quick Test with Streamlit (Easiest)

```bash
# Run the web app
streamlit run app/streamlit_app.py
```

Then:
1. Use the app to upload test images
2. Or use GEE mode to download fresh data

### Option 2: Train Models with Sen1Floods11

**Step 1**: Download small dataset (~10-15 images, 1-2 GB)
- Visit: https://github.com/cloudtostreet/Sen1Floods11
- Select ~10-15 sample flood images
- Download and organize:
  ```
  data/raw/S1/     <- Sentinel-1 SAR
  data/raw/S2/     <- Sentinel-2 optical  
  data/raw/labels/ <- Flood masks
  ```

**Step 2**: Train models
```bash
# Train ResNet
python src/training/train.py --model resnet

# Train Swin Transformer
python src/training/train.py --model swin

# Train MaxViT
python src/training/train.py --model maxvit
```

**Step 3**: Test with trained models
```bash
streamlit run app/streamlit_app.py
```

### Option 3: Use Google Colab (For Easy Setup)

1. Create new Colab notebook
2. Install geemap and earthengine
3. Download data there (easier than Windows)
4. Transfer data to your local machine

## File Structure

```
Flood detection/
├── src/              # All code (ready to use)
│   ├── data/         # Data loading, preprocessing
│   ├── models/       # Model architectures
│   ├── training/     # Training scripts
│   └── inference/     # Inference pipeline
├── app/              # Streamlit web app
├── configs/          # Configuration file
├── models/           # (Will contain trained models)
└── data/             # (Will contain dataset)
```

## What You Have Now

- ✅ Complete codebase for flood detection
- ✅ All model architectures implemented
- ✅ Training pipeline ready
- ✅ Streamlit web application
- ✅ Google Earth Engine integration

## What You Need

- Data (~1-2 GB of Sen1Floods11 or use GEE)
- (Optional) GPU for faster training

## Commands Reference

```bash
# View dataset options
python download_sample_data.py

# Train a model
python src/training/train.py --model resnet

# Run web app
streamlit run app/streamlit_app.py

# View all options
python download_sample_data.py
```

## Troubleshooting

**Problem**: Windows GDAL/rasterio issues
**Solution**: Use Option 1 (Streamlit app) or Option 3 (Google Colab)

**Problem**: No data to train on
**Solution**: Download Sen1Floods11 or use Streamlit GEE mode

**Problem**: Models not found
**Solution**: Train models first with `train.py`

## Next: Start with Streamlit App

The easiest way to get started:

```bash
streamlit run app/streamlit_app.py
```

This gives you a working flood detection interface!

