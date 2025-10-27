# Flood Detection System - Implementation Summary

## Project Overview

A complete flood detection system using deep learning ensemble models with Sentinel-1 SAR and Sentinel-2 optical data fusion.

## What Has Been Implemented

### ✅ Core Components

1. **Project Structure** (`setup-environment`)
   - Created modular folder organization
   - Set up data, models, src, app directories
   - Created configuration files and documentation

2. **Google Earth Engine Integration** (`gee-integration`)
   - `src/data/gee_downloader.py`: Automated S1 and S2 data fetching
   - Spatial alignment and co-registration
   - Filtering for cloud coverage and quality

3. **Data Pipeline** (`data-pipeline`)
   - `src/data/dataset.py`: PyTorch Dataset class for flood data
   - `src/data/preprocessing.py`: Alignment, normalization, quality control
   - Train/val/test splitting functionality

4. **Model Architectures** (`model-resnet`, `model-swin`, `model-maxvit`)
   - `src/models/resnet_unet.py`: ResNet-50 UNet for flood segmentation
   - `src/models/swin_transformer.py`: Swin Transformer architecture
   - `src/models/max_vit.py`: MaxViT hybrid CNN-ViT model
   - All models adapted for multi-channel input (S1 + S2)

5. **Ensemble & Uncertainty** (`ensemble-uncertainty`)
   - `src/models/ensemble.py`: Model combination with uncertainty quantification
   - Epistemic and aleatoric uncertainty calculation
   - Model soups and prediction aggregation

6. **Training Pipeline** (`training-pipeline`)
   - `src/training/train.py`: Complete training script
   - `src/training/losses.py`: Dice, Focal, Combined loss functions
   - `src/training/metrics.py`: IoU, F1, Precision, Recall calculations
   - Early stopping and checkpointing

7. **Inference Pipeline** (`inference-pipeline`)
   - `src/inference/predict.py`: Flood prediction API
   - `src/inference/uncertainty.py`: Uncertainty quantification
   - Support for both GeoTIFF and numpy array inputs

8. **Streamlit Application** (`streamlit-backend`, `streamlit-ui`)
   - `app/streamlit_app.py`: Complete web interface
   - `app/utils.py`: Visualization and utility functions
   - Upload mode and Google Earth Engine mode
   - Interactive visualization with uncertainty maps

9. **Documentation** (`documentation`)
   - `README.md`: Comprehensive project documentation
   - `SETUP.md`: Installation and setup guide
   - All code with docstrings and type hints

## File Structure

```
flood-detection/
├── src/
│   ├── data/
│   │   ├── dataset.py           # PyTorch Dataset
│   │   ├── preprocessing.py     # Data preprocessing
│   │   ├── gee_downloader.py    # Google Earth Engine
│   │   └── __init__.py
│   ├── models/
│   │   ├── resnet_unet.py       # ResNet-50 UNet
│   │   ├── swin_transformer.py  # Swin Transformer
│   │   ├── max_vit.py           # MaxViT
│   │   ├── ensemble.py          # Ensemble module
│   │   └── __init__.py
│   ├── training/
│   │   ├── train.py             # Training script
│   │   ├── losses.py            # Loss functions
│   │   ├── metrics.py           # Evaluation metrics
│   │   └── __init__.py
│   └── inference/
│       ├── predict.py           # Inference API
│       ├── uncertainty.py       # Uncertainty quantification
│       └── __init__.py
├── app/
│   ├── streamlit_app.py         # Web application
│   ├── utils.py                 # App utilities
│   └── __init__.py
├── configs/
│   └── config.yaml              # Configuration
├── notebooks/                   # Jupyter notebooks
├── data/                        # Data directories
├── models/                      # Trained models
├── requirements.txt             # Dependencies
├── README.md                    # Main documentation
├── SETUP.md                     # Setup guide
└── .gitignore                   # Git ignore rules
```

## Key Features

### 1. Multi-Model Ensemble
- ResNet-50 UNet with pretrained encoder
- Swin Transformer for global context
- MaxViT for multi-scale features
- Ensembled predictions with uncertainty

### 2. Data Fusion
- S1 (2 bands: VV, VH) + S2 (4 bands: RGB+NIR) = 6 channels
- Spatial alignment and normalization
- Quality control and preprocessing

### 3. Uncertainty Quantification
- Epistemic uncertainty (model disagreement)
- Aleatoric uncertainty (data uncertainty)
- Confidence maps for decision-making

### 4. Automated Pipeline
- Google Earth Engine integration
- Preprocessing and augmentation
- Batch processing support

### 5. User-Friendly Interface
- Streamlit web application
- Upload or GEE download options
- Interactive visualization
- Statistics and download options

## Usage Examples

### Training
```bash
python src/training/train.py --model resnet
python src/training/train.py --model swin
python src/training/train.py --model maxvit
```

### Inference
```python
from src.inference.predict import FloodPredictor
predictor = FloodPredictor(model_paths)
pred, probs = predictor.predict_from_tif(s1_path, s2_path)
```

### Web App
```bash
streamlit run app/streamlit_app.py
```

## Remaining Tasks

1. **Model Training** (`train-models`)
   - Download Sen1Floods11 dataset
   - Train all three models
   - Achieve target IoU > 0.72

2. **Validation & Testing** (`validation-testing`)
   - Test on held-out test set
   - Real-world flood event validation
   - End-to-end pipeline testing

## Next Steps

1. Download Sen1Floods11 dataset from https://github.com/cloudtostreet/Sen1Floods11
2. Organize data in `data/raw/S1/`, `data/raw/S2/`, `data/raw/labels/`
3. Train models using the training scripts
4. Test the web application
5. Validate on real flood events

## Technical Highlights

- **Architecture**: Hybrid CNN-ViT ensemble
- **Input**: Multi-channel satellite imagery (6 bands)
- **Output**: Binary flood masks with uncertainty maps
- **Target Metrics**: IoU > 0.72, F1 > 0.80
- **Framework**: PyTorch + Streamlit
- **Data Sources**: Sen1Floods11 + Google Earth Engine

## Dependencies

See `requirements.txt` for complete list. Key packages:
- PyTorch, torchvision, timm
- rasterio, geopandas, geemap
- streamlit, streamlit-folium
- numpy, matplotlib, pandas

## Credits

Based on DeepSARFlood (Sharma & Saharia, 2025)
Repository: https://github.com/hydrosenselab/DeepSARFlood

