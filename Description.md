# Flood Detection Project – Comprehensive Description

## Executive Summary
This project builds a practical, end‑to‑end flood detection system using satellite imagery. It fuses Sentinel‑1 SAR (cloud‑penetrating) and Sentinel‑2 optical data, trains deep learning models for pixel‑wise flood segmentation, and provides an application‑focused interface via Streamlit. Training and testing are performed on Google Colab to avoid local environment issues (notably PyTorch DLL and GDAL on Windows). The final workflow is reproducible, lightweight (1–2 GB data subset), and presentation‑ready with clear visuals and metrics.

- Objective: Accurate flood inundation mapping with small data footprint and fast turnaround
- Core stack: PyTorch, Rasterio, Streamlit, Google Colab, Google Drive, optional GEE
- Models: Simple UNet‑style CNN (MVP) with roadmap for ResNet‑UNet, Swin Transformer, and MaxViT; ensemble and uncertainty planned
- Deployment: Local Streamlit dashboard (status/docs) + Colab notebooks/scripts for training/testing

---

## What We Built (Non‑Technical Overview)
- Data pipeline to prepare small subsets of the Sen1Floods11 dataset (SAR + labels)
- Training scripts for Colab that automatically install dependencies, handle Drive, and train a segmentation network
- Robust testing scripts that generate prediction maps and publication‑quality charts directly in Colab
- A Streamlit dashboard to guide usage, explain status, and avoid local GPU dependencies
- Clear documentation, troubleshooting, and a reproducible workflow for demonstration

Why this matters:
- SAR works in all weather conditions; ideal for floods with cloud cover
- A small, efficient prototype lowers compute barriers while retaining real‑world utility
- Colab execution avoids OS‑specific installation headaches

---

## End‑to‑End Process (What We Did)
1. Planning from MVP and DeepSARFlood paper; defined a small‑data, Colab‑first track.
2. Implemented data pipeline and minimal model to establish training loop and outputs.
3. Pivoted away from local heavy dependencies (PyTorch/GDAL on Windows) to Colab.
4. Created robust training and testing scripts that:
   - Mount Drive, copy data
   - Install correct libraries (torch, rasterio)
   - Read GeoTIFFs safely
   - Train model with class‑imbalance‑aware loss
   - Save and download best checkpoints
5. Added visualization: side‑by‑side predictions, overlays, error maps, and metric distributions.
6. Built a Streamlit dashboard for local guidance without GPU or torch.
7. Documented issues, fixes, and instructions for presentation and reuse.

---

## Results (Current MVP)
- Successful Colab training pipeline with robust data handling (NaN/Inf checks, clipping, gradient clipping)
- Testing script generates:
  - Example predictions (SAR, ground truth, prediction, overlay, error maps)
  - IoU/Precision/Recall/F1 distributions and summary
- If initial models underperform (e.g., class imbalance), scripts guide weighted loss retraining and diagnostics
- Final artifacts automatically download: best checkpoint and all plots

Note: Achieving IoU ≥ 0.72 typically requires more data, longer training, and larger backbones (e.g., ResNet‑UNet or ViT variants). The current pipeline is designed to scale to that.

---

## Key Challenges and Fixes
- Windows PyTorch DLL (WinError 1114): Migrated training and inference to Colab.
- GDAL/Rasterio on Windows: Avoided local install; use `rasterio` in Colab where it works reliably.
- TIFF reading errors with PIL: Switched to `rasterio` for GeoTIFFs.
- Class imbalance (flood ≪ background): Added class‑weighted CrossEntropy and optional focal‑style training.
- NaN training instability: Introduced value clipping, per‑channel normalization guards, gradient clipping, and NaN/batch skipping logic.
- Checkpoint loading (`weights_only=True` default in PyTorch 2.6): Explicitly set `weights_only=False`.
- Colab directory overwrite errors: Safely remove destination folders before copying.

---

## How to Present the Project (Demo Script)
1. Open Colab, run `COMPLETE_FLOOD_DETECTION_COLAB.py`.
2. Show training logs (loss + IoU improving) and resulting plots as they render.
3. Show saved figures:
   - Training/validation curves and metric trends
   - Predictions grid with overlays and error maps
   - Metric distributions and summary panel
4. Download the model and results; briefly open images to highlight insights.
5. Locally, run the Streamlit dashboard as a guided tour of the project.

---

## File‑by‑File Explanation (What each file does)

### Top‑Level
- `README.md`: Main project documentation (features, setup, usage, structure, performance, troubleshooting, citations).
- `requirements.txt`: Python dependencies for local environments (not used for Colab).
- `.gitignore`: Excludes build artifacts, virtual envs, datasets, large checkpoints, etc.
- `setup.py`: Optional packaging helper for the project.
- `PROJECT_SUMMARY.md`: Implementation summary and next steps (concise).
- `TESTING_GUIDE.md`: Focused instructions for testing trained models in Colab.
- `QUICK_START.md`: Short, task‑oriented getting started flow.
- `STATUS.md`: Running status updates and milestones.
- `deepsar.pdf`: The referenced DeepSARFlood paper for context/background.

### Configuration
- `configs/config.yaml`: Central configuration for training/inference hyperparameters, data, and ensemble weights. Useable by future expanded pipeline.

### Source Code (`src/`)
- `src/data/gee_downloader.py`: Google Earth Engine integration utility (class `GEEDownloader`) for S1/S2 retrieval and alignment. Not required for Colab MVP, but part of full pipeline.
- `src/data/preprocessing.py`: Helper transforms for stacking, normalization, and alignment.
- `src/data/dataset.py`: General PyTorch dataset class for flood segmentation; handles reading, transforms, and augmentation.
- `src/models/resnet_unet.py`: ResNet‑50 UNet segmentation model.
- `src/models/swin_transformer.py`: Swin Transformer‑based segmentation.
- `src/models/max_vit.py`: MaxViT hybrid model.
- `src/models/ensemble.py`: Model soup/ensemble utilities and uncertainty estimation.
- `src/training/losses.py`: Dice, BCE, and composite losses.
- `src/training/metrics.py`: IoU, Precision, Recall, F1 utilities.
- `src/training/train.py`: Canonical training entry‑point (config‑driven), used for larger training jobs.
- `src/inference/predict.py`: Inference helper (`FloodPredictor`) for batch/CLI/API use.
- `src/inference/uncertainty.py`: Epistemic/aleatoric uncertainty helpers.

### Streamlit App (`app/`)
- `app/streamlit_app.py`: Full app (requires torch) – designed for systems without PyTorch limitations.
- `app/streamlit_app_simple.py`: Lightweight dashboard without torch; runs locally on Windows.
- `app/utils.py`: Shared app utilities for plotting/formatting/geospatial helpers.

### Colab‑First Scripts (Core for MVP)
- `COLAB_TRAINING_SINGLE_CELL.py`: Minimalistic, single‑cell training script for small datasets; establishes pipeline.
- `TEST_MODEL_COLAB.py`: Testing/inference in Colab; loads a model, generates predictions and metrics, and saves plots.
- `DIAGNOSE_AND_FIX_MODEL.py`: Diagnostics for label coverage, model output sanity checks, and a weighted‑loss retraining option. Fixes `weights_only` and Drive copy issues.
- `ROBUST_RETRAIN_COLAB.py`: Stronger training loop with NaN/Inf detection, clipping, normalization guards, and gradient clipping; saves best checkpoint and visuals.
- `COMPLETE_FLOOD_DETECTION_COLAB.py`: All‑in‑one training + evaluation + visualization + downloads. Presentation‑ready, generates: training curves, predictions grid, and summary charts.

### Notebooks
- `download_sen1floods11_colab.ipynb`: Google Cloud Storage/gsutil‑based subset downloader for Sen1Floods11.
- `train_flood_detection_colab.ipynb`: Notebook variant for training (alternative to scripts).

### Data Layout
- `data/raw/S1/*.tif`: Sentinel‑1 SAR GeoTIFF tiles (VV/VH)
- `data/raw/S2/*.tif`: Sentinel‑2 optical tiles (optional for MVP)
- `data/raw/labels/*.tif`: Binary flood masks aligned with S1/S2
- `data/processed/`: Preprocessed tensors/tiles (optional)

---

## How to Reproduce (Step‑by‑Step)

### A) Training & Visualization in Colab (Recommended)
1. Open Colab → new Python 3 notebook.
2. Copy‑paste contents of `COMPLETE_FLOOD_DETECTION_COLAB.py` into a single cell.
3. Run the cell.
4. Upload/ensure dataset at Drive path: `MyDrive/Flood detection/data/raw/{S1,labels}`.
5. Observe:
   - Real‑time training logs for 30 epochs
   - Plots inline: loss, IoU/metrics, prediction samples, distributions, summary panel
   - Automatic downloads: `flood_model_BEST.pt` + three PNGs + `RESULTS_SUMMARY.txt`

If you only want to test an existing model, use `TEST_MODEL_COLAB.py` instead.

### B) Local Dashboard
- Run: `python -m streamlit run app/streamlit_app_simple.py`
- Open `http://localhost:8501` for a status/instructions dashboard (no torch required).

---

## Technical Design Notes
- Input shape: 2‑channel S1 (VV/VH) resized to 256×256 for the MVP scripts
- Normalization: Per‑channel z‑score with safety guards; clip extremes to avoid NaN
- Loss: Class‑weighted CrossEntropy to handle rare flood pixels
- Optimizer: AdamW; Scheduler: CosineAnnealingLR; Gradient clipping to stabilize
- Metrics: IoU, Precision, Recall, F1 computed per image and averaged
- Checkpoint policy: Save best by validation IoU; fall back to last if none improved

---

## Troubleshooting Quick Reference
- PIL can’t read TIFF → Use `rasterio` (handled in Colab scripts).
- `UnpicklingError` on torch.load → pass `weights_only=False`.
- `FileExistsError` when copying data in Colab → remove destination folder before `copytree`.
- All‑zeros predictions → run `DIAGNOSE_AND_FIX_MODEL.py`; retrain with class weights.
- NaN loss during training → use `ROBUST_RETRAIN_COLAB.py` (clipping + NaN guards).
- Streamlit can’t import torch on Windows → use `app/streamlit_app_simple.py` locally.

---

## Limitations and Future Work
- Current minimal model is intentionally lightweight for MVP. To reach ≥0.72 IoU:
  - Switch to ResNet‑50 UNet / Swin / MaxViT pipelines in `src/models/`
  - Increase training epochs and data volume (full Sen1Floods11)
  - Add data augmentation and multi‑temporal inputs
  - Implement deep ensembles and uncertainty maps end‑to‑end in the app
- Integrate GEE fetch inside the app when running on a Linux server or cloud notebook.

---

## Presentation Tips
- Start with the problem (flood mapping), constraints (weather/clouds), and why SAR.
- Show the training pipeline screenshot/logs from Colab.
- Display the prediction grid (SAR, GT, pred, overlay, error map) and metric charts.
- Explain how class imbalance was addressed and why NaN guards were needed.
- End with roadmap: larger backbones, ensembles, uncertainty, and full GEE automation.

---

## Credits
- Research reference: DeepSARFlood (Vision Transformer ensembles for SAR flood mapping)
- Dataset: Sen1Floods11 (Cloud to Street)
- Tools: PyTorch, Rasterio, Streamlit, Google Colab/Drive
