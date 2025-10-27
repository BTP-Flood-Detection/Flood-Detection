"""
TEST YOUR TRAINED FLOOD DETECTION MODEL
Copy this entire script into a Google Colab cell and run it.

This will:
1. Load your trained model
2. Test on validation/test images
3. Show predictions vs ground truth
4. Calculate accuracy metrics (IoU, F1, etc.)
5. Visualize results
"""

# ============================================================================
# STEP 1: Setup and Mount Drive
# ============================================================================
print("📂 Mounting Google Drive...")
from google.colab import drive
drive.mount('/content/drive')

import os
import shutil

# Copy data from Drive
print("📥 Copying test data...")
src_data = '/content/drive/MyDrive/Flood detection/data'
dst_data = '/content/test_data'

# Remove existing folder if it exists
if os.path.exists(dst_data):
    shutil.rmtree(dst_data)
    print("🗑️ Removed old data")

if os.path.exists(src_data):
    shutil.copytree(src_data, dst_data)
    print(f"✅ Data copied")
else:
    print("⚠️ Upload your data to Google Drive first")

# ============================================================================
# STEP 2: Install Dependencies
# ============================================================================
print("\n📦 Installing dependencies...")
import subprocess
import sys

packages = ["torch", "torchvision", "matplotlib", "scikit-learn", "tqdm", "rasterio"]
for package in packages:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])

print("✅ Dependencies installed")

# ============================================================================
# STEP 3: Import Libraries
# ============================================================================
import torch
import torch.nn as nn
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import torchvision.transforms as T

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n🖥️ Using device: {device}")

# ============================================================================
# STEP 4: Define Model Architecture (Same as Training)
# ============================================================================
class SimpleFloodModel(nn.Module):
    def __init__(self, input_channels=2, num_classes=2):
        super().__init__()
        
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.pool2 = nn.MaxPool2d(2)
        
        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Decoder
        self.up1 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.final = nn.Conv2d(64, num_classes, 1)
    
    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        
        e3 = self.enc3(p2)
        
        u1 = self.up1(e3)
        d1 = torch.cat([u1, e2], dim=1)
        d1 = self.dec1(d1)
        
        u2 = self.up2(d1)
        d2 = torch.cat([u2, e1], dim=1)
        d2 = self.dec2(d2)
        
        out = self.final(d2)
        return out

# ============================================================================
# STEP 5: Load Your Trained Model
# ============================================================================
print("\n📥 Upload your trained model...")
print("Click the folder icon on the left → Upload 'flood_model_best.pt'")
print("Or use the file upload below:")

from google.colab import files
uploaded = files.upload()

# Get the uploaded file name
model_file = list(uploaded.keys())[0]
print(f"\n✅ Model uploaded: {model_file}")

# Load model
print("🔄 Loading model...")
model = SimpleFloodModel(input_channels=2, num_classes=2).to(device)
checkpoint = torch.load(model_file, map_location=device, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"✅ Model loaded! Best IoU from training: {checkpoint.get('iou', 'N/A'):.4f}")

# ============================================================================
# STEP 6: Prepare Test Data
# ============================================================================
print("\n📊 Preparing test data...")

def load_and_preprocess(s1_path, label_path=None):
    """Load and preprocess image using rasterio for geospatial TIFFs"""
    try:
        # Load S1 using rasterio
        with rasterio.open(s1_path) as src:
            s1_array = src.read()  # Read all bands (C, H, W)
            s1_array = s1_array.astype(np.float32)
        
        # Handle channels - take first 2 bands (VV, VH)
        if s1_array.shape[0] >= 2:
            s1_array = s1_array[:2]  # Take first 2 channels
        elif s1_array.shape[0] == 1:
            s1_array = np.stack([s1_array[0], s1_array[0]], axis=0)  # Duplicate channel
        
        # Convert to tensor (already in C, H, W format)
        s1_tensor = torch.from_numpy(s1_array)
        
        # Resize
        target_size = (256, 256)
        s1_tensor = T.functional.resize(s1_tensor, target_size)
        
        # Normalize
        s1_tensor = (s1_tensor - s1_tensor.mean()) / (s1_tensor.std() + 1e-8)
        
        # Load label if provided
        if label_path:
            with rasterio.open(label_path) as src:
                label_array = src.read(1)  # Read first band
                label_array = label_array.astype(np.float32)
            
            # Binary classification: flood (1) vs no-flood (0)
            label_array = (label_array > 0).astype(np.float32)
            
            label_tensor = torch.from_numpy(label_array).unsqueeze(0)
            label_tensor = T.functional.resize(label_tensor, target_size)
            label_tensor = label_tensor.squeeze(0).long()
            
            return s1_tensor, label_tensor
        
        return s1_tensor, None
        
    except Exception as e:
        print(f"Error loading {s1_path}: {e}")
        return None, None

# Get test images
s1_dir = Path(f"{dst_data}/raw/S1")
labels_dir = Path(f"{dst_data}/raw/labels")

test_files = sorted(list(s1_dir.glob("*.tif")))[-20:]  # Last 20 images for testing
label_files = sorted(list(labels_dir.glob("*.tif")))[-20:]

print(f"Testing on {len(test_files)} images")

# ============================================================================
# STEP 7: Run Predictions and Calculate Metrics
# ============================================================================
print("\n🔮 Running predictions...")

def calculate_metrics(pred, target):
    """Calculate IoU, F1, Precision, Recall"""
    pred = pred.flatten()
    target = target.flatten()
    
    # IoU
    intersection = ((pred == 1) & (target == 1)).sum().item()
    union = ((pred == 1) | (target == 1)).sum().item()
    iou = intersection / union if union > 0 else 0
    
    # Precision, Recall, F1
    tp = ((pred == 1) & (target == 1)).sum().item()
    fp = ((pred == 1) & (target == 0)).sum().item()
    fn = ((pred == 0) & (target == 1)).sum().item()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'iou': iou,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

all_metrics = []
predictions = []
ground_truths = []
images = []
skipped_count = 0

with torch.no_grad():
    for s1_file, label_file in tqdm(zip(test_files, label_files), total=len(test_files)):
        image, label = load_and_preprocess(str(s1_file), str(label_file))
        
        if image is None or label is None:
            skipped_count += 1
            continue
        
        # Predict
        image_batch = image.unsqueeze(0).to(device)
        output = model(image_batch)
        pred = output.argmax(dim=1).squeeze(0).cpu()
        
        # Calculate metrics
        metrics = calculate_metrics(pred, label)
        all_metrics.append(metrics)
        
        # Store for visualization
        predictions.append(pred.numpy())
        ground_truths.append(label.numpy())
        images.append(image.cpu().numpy())

print(f"\n✅ Successfully processed {len(all_metrics)} images")
if skipped_count > 0:
    print(f"⚠️ Skipped {skipped_count} corrupted/unreadable images")

# ============================================================================
# STEP 8: Display Results
# ============================================================================
print("\n" + "="*60)
print("📊 TEST RESULTS")
print("="*60)

if len(all_metrics) == 0:
    print("❌ ERROR: No images were successfully processed!")
    print("Please check your data files. They may be corrupted or in the wrong format.")
    print("="*60)
else:
    # Average metrics
    avg_metrics = {
        'IoU': np.mean([m['iou'] for m in all_metrics]),
        'Precision': np.mean([m['precision'] for m in all_metrics]),
        'Recall': np.mean([m['recall'] for m in all_metrics]),
        'F1 Score': np.mean([m['f1'] for m in all_metrics])
    }

    for metric, value in avg_metrics.items():
        print(f"{metric:15s}: {value:.4f}")

    print("="*60)

    # Check if target met
    if avg_metrics['IoU'] > 0.72:
        print("🎉 ✅ TARGET MET! IoU > 0.72")
    else:
        print(f"⚠️ Target not met. Current: {avg_metrics['IoU']:.4f}, Target: 0.72")

# ============================================================================
# STEP 9: Visualize Predictions
# ============================================================================
if len(predictions) > 0:
    print("\n📊 Generating visualizations...")

    # Show 6 random examples
    num_samples = min(6, len(predictions))
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, num_samples * 3.5))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    fig.suptitle('Flood Detection Results', fontsize=16)

    indices = np.random.choice(len(predictions), num_samples, replace=False)

    for idx, sample_idx in enumerate(indices):
        # Original image (first channel)
        axes[idx, 0].imshow(images[sample_idx][0], cmap='gray')
        axes[idx, 0].set_title('S1 SAR Image')
        axes[idx, 0].axis('off')
        
        # Ground truth
        axes[idx, 1].imshow(ground_truths[sample_idx], cmap='Blues')
        axes[idx, 1].set_title('Ground Truth')
        axes[idx, 1].axis('off')
        
        # Prediction
        axes[idx, 2].imshow(predictions[sample_idx], cmap='Blues')
        metrics = all_metrics[sample_idx]
        axes[idx, 2].set_title(f"Prediction (IoU: {metrics['iou']:.3f})")
        axes[idx, 2].axis('off')

    plt.tight_layout()
    plt.savefig('flood_predictions.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("✅ Visualization saved as 'flood_predictions.png'")
else:
    print("\n⚠️ No predictions to visualize")

# ============================================================================
# STEP 10: Detailed Metrics Plot
# ============================================================================
if len(all_metrics) > 0:
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    metrics_data = {
        'IoU': [m['iou'] for m in all_metrics],
        'Precision': [m['precision'] for m in all_metrics],
        'Recall': [m['recall'] for m in all_metrics],
        'F1 Score': [m['f1'] for m in all_metrics]
    }

    for idx, (metric_name, values) in enumerate(metrics_data.items()):
        row, col = idx // 2, idx % 2
        axes[row, col].hist(values, bins=20, edgecolor='black', alpha=0.7)
        axes[row, col].axvline(np.mean(values), color='red', linestyle='--', 
                               label=f'Mean: {np.mean(values):.3f}')
        axes[row, col].set_xlabel(metric_name)
        axes[row, col].set_ylabel('Frequency')
        axes[row, col].set_title(f'{metric_name} Distribution')
        axes[row, col].legend()

    plt.tight_layout()
    plt.savefig('metrics_distribution.png', dpi=150)
    plt.show()

    print("✅ Metrics distribution saved as 'metrics_distribution.png'")
else:
    print("⚠️ No metrics to plot")

# ============================================================================
# STEP 11: Download Results
# ============================================================================
if len(all_metrics) > 0:
    print("\n📥 Download results...")

    # Download images
    files.download('flood_predictions.png')
    files.download('metrics_distribution.png')

    print("\n" + "="*60)
    print("✅ TESTING COMPLETE!")
    print("="*60)
    print("\nYour model performance:")
    for metric, value in avg_metrics.items():
        print(f"  {metric}: {value:.4f}")
    print("\nImages downloaded. Check your Downloads folder!")
    print("="*60)
else:
    print("\n" + "="*60)
    print("❌ TESTING FAILED")
    print("="*60)
    print("No valid test data could be processed.")
    print("Please check that your data files are valid GeoTIFF images.")
    print("="*60)

