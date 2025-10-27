# Setup Guide

Quick start guide for the Flood Detection System.

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 2: Google Earth Engine Setup

1. Sign up for a Google Earth Engine account: https://earthengine.google.com/

2. Authenticate:
```bash
earthengine authenticate
```

Follow the instructions to complete authentication.

## Step 3: Download Sen1Floods11 Dataset

1. Visit: https://github.com/cloudtostreet/Sen1Floods11

2. Download the dataset or access via the repository

3. Organize data in the following structure:
```
data/raw/S1/     # Sentinel-1 images
data/raw/S2/     # Sentinel-2 images  
data/raw/labels/ # Flood masks
```

## Step 4: Train Models (Optional)

If you want to train models from scratch:

```bash
# Train ResNet model
python src/training/train.py --model resnet

# Train Swin Transformer
python src/training/train.py --model swin

# Train MaxViT
python src/training/train.py --model maxvit
```

Training may take several hours depending on your GPU.

## Step 5: Run the Web Application

```bash
streamlit run app/streamlit_app.py
```

The app will open in your browser at `http://localhost:8501`

## Using Pre-trained Models

If you have pre-trained models:
1. Place them in the `models/` directory
2. Name them: `resnet_best.pt`, `swin_best.pt`, `maxvit_best.pt`
3. The Streamlit app will automatically load them

## Troubleshooting

### Issue: Earth Engine authentication fails
- Make sure you have a GEE account
- Run `earthengine authenticate` and follow the browser prompt

### Issue: CUDA out of memory
- Reduce batch size in `configs/config.yaml`
- Set `batch_size` to 8 or 4

### Issue: Models not found
- Train models first or download pre-trained weights
- Check that model files are in the `models/` directory

## Next Steps

1. Explore the notebooks in `notebooks/` for data exploration
2. Test on your own flood images
3. Adjust model configurations in `configs/config.yaml`

