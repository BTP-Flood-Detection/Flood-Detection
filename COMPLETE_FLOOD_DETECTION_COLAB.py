"""
🌊 COMPLETE FLOOD DETECTION - TRAIN & TEST WITH VISUALIZATIONS
Copy this entire script into Google Colab and run it!

This will:
1. ✅ Load and validate your data
2. ✅ Train a robust flood detection model (30 epochs)
3. ✅ Test on validation set
4. ✅ Generate beautiful visualizations and graphs
5. ✅ Download everything (model + all plots)
"""

print("="*70)
print("🌊 FLOOD DETECTION - COMPLETE PIPELINE")
print("="*70)

# ============================================================================
# STEP 1: Setup
# ============================================================================
print("\n📂 Mounting Google Drive...")
from google.colab import drive
drive.mount('/content/drive')

import os
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Set style for better plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Install dependencies
print("\n📦 Installing dependencies...")
import subprocess
import sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "rasterio", "seaborn"])

import rasterio
import torchvision.transforms as T

# Copy data
print("\n📥 Copying data from Google Drive...")
src_data = '/content/drive/MyDrive/Flood detection/data'
dst_data = '/content/flood_data'

if os.path.exists(dst_data):
    shutil.rmtree(dst_data)

shutil.copytree(src_data, dst_data)
print("✅ Data copied successfully!")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n🖥️ Using device: {device}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")

# ============================================================================
# STEP 2: Robust Dataset Class
# ============================================================================
class RobustFloodDataset(Dataset):
    def __init__(self, s1_files, label_files, validate=True):
        self.s1_files = s1_files
        self.label_files = label_files
        
        if validate:
            print("🔍 Validating data files...")
            self.valid_indices = []
            for idx in tqdm(range(len(s1_files))):
                if self._validate_sample(idx):
                    self.valid_indices.append(idx)
            print(f"✅ Valid samples: {len(self.valid_indices)} / {len(s1_files)}")
        else:
            self.valid_indices = list(range(len(s1_files)))
    
    def _validate_sample(self, idx):
        try:
            with rasterio.open(str(self.s1_files[idx])) as src:
                s1 = src.read().astype(np.float32)
            with rasterio.open(str(self.label_files[idx])) as src:
                label = src.read(1).astype(np.float32)
            
            if np.isnan(s1).any() or np.isinf(s1).any():
                return False
            if np.isnan(label).any() or np.isinf(label).any():
                return False
            if s1.sum() == 0:
                return False
            return True
        except:
            return False
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        
        try:
            with rasterio.open(str(self.s1_files[real_idx])) as src:
                s1 = src.read().astype(np.float32)
            
            if s1.shape[0] >= 2:
                s1 = s1[:2]
            elif s1.shape[0] == 1:
                s1 = np.stack([s1[0], s1[0]], axis=0)
            
            with rasterio.open(str(self.label_files[real_idx])) as src:
                label = src.read(1).astype(np.float32)
            
            label = (label > 0).astype(np.float32)
            
            s1_tensor = torch.from_numpy(s1)
            s1_tensor = T.functional.resize(s1_tensor, (256, 256))
            
            label_tensor = torch.from_numpy(label).unsqueeze(0)
            label_tensor = T.functional.resize(label_tensor, (256, 256))
            label_tensor = label_tensor.squeeze(0).long()
            
            # Robust normalization
            s1_tensor = torch.clamp(s1_tensor, -100, 100)
            for c in range(s1_tensor.shape[0]):
                channel = s1_tensor[c]
                mean = channel.mean()
                std = channel.std()
                if std < 1e-6:
                    std = 1.0
                s1_tensor[c] = (channel - mean) / std
            s1_tensor = torch.clamp(s1_tensor, -10, 10)
            
            if torch.isnan(s1_tensor).any():
                s1_tensor = torch.zeros_like(s1_tensor)
            
            return s1_tensor, label_tensor
        except:
            return torch.zeros(2, 256, 256), torch.zeros(256, 256).long()

# ============================================================================
# STEP 3: Model Architecture
# ============================================================================
class FloodDetectionModel(nn.Module):
    def __init__(self, input_channels=2, num_classes=2):
        super().__init__()
        
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
# STEP 4: Prepare Data
# ============================================================================
print("\n📊 Preparing datasets...")
s1_dir = Path(f"{dst_data}/raw/S1")
labels_dir = Path(f"{dst_data}/raw/labels")

s1_all = sorted(list(s1_dir.glob("*.tif")))
label_all = sorted(list(labels_dir.glob("*.tif")))

print(f"Found {len(s1_all)} S1 images and {len(label_all)} labels")

split_idx = int(0.8 * len(s1_all))
train_dataset = RobustFloodDataset(s1_all[:split_idx], label_all[:split_idx], validate=True)
val_dataset = RobustFloodDataset(s1_all[split_idx:], label_all[split_idx:], validate=False)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2)

print(f"✅ Training: {len(train_dataset)} samples, Validation: {len(val_dataset)} samples")

# ============================================================================
# STEP 5: Training Setup
# ============================================================================
print("\n⚙️ Initializing model and training...")

model = FloodDetectionModel(input_channels=2, num_classes=2).to(device)
optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.01)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

# Weighted loss for class imbalance
class_weights = torch.tensor([1.0, 5.0], dtype=torch.float32).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)

print("✅ Model ready!")
print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

# ============================================================================
# STEP 6: Training Loop
# ============================================================================
print("\n🚀 Starting training (30 epochs)...")
print("="*70)

best_iou = 0
best_epoch = 0
history = {
    'train_loss': [], 'val_loss': [], 'val_iou': [],
    'val_precision': [], 'val_recall': [], 'val_f1': []
}

for epoch in range(30):
    # Train
    model.train()
    train_loss = 0
    nan_count = 0
    
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1:02d}/30"):
        images, labels = images.to(device), labels.to(device)
        
        if torch.isnan(images).any():
            nan_count += 1
            continue
        
        optimizer.zero_grad()
        outputs = model(images)
        
        if torch.isnan(outputs).any():
            nan_count += 1
            continue
        
        loss = criterion(outputs, labels)
        
        if torch.isnan(loss):
            nan_count += 1
            continue
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_loss += loss.item()
    
    train_loss /= max(len(train_loader) - nan_count, 1)
    
    # Validate
    model.eval()
    val_loss = 0
    all_iou, all_precision, all_recall, all_f1 = [], [], [], []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            
            if torch.isnan(images).any():
                continue
            
            outputs = model(images)
            
            if torch.isnan(outputs).any():
                continue
            
            loss = criterion(outputs, labels)
            if not torch.isnan(loss):
                val_loss += loss.item()
            
            preds = outputs.argmax(dim=1)
            
            # Calculate metrics per image
            for pred, label in zip(preds, labels):
                tp = ((pred == 1) & (label == 1)).sum().item()
                fp = ((pred == 1) & (label == 0)).sum().item()
                fn = ((pred == 0) & (label == 1)).sum().item()
                
                intersection = tp
                union = tp + fp + fn
                iou = intersection / union if union > 0 else 0
                all_iou.append(iou)
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                all_precision.append(precision)
                all_recall.append(recall)
                all_f1.append(f1)
    
    val_loss /= max(len(val_loader), 1)
    val_iou = np.mean(all_iou) if all_iou else 0
    val_precision = np.mean(all_precision) if all_precision else 0
    val_recall = np.mean(all_recall) if all_recall else 0
    val_f1 = np.mean(all_f1) if all_f1 else 0
    
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['val_iou'].append(val_iou)
    history['val_precision'].append(val_precision)
    history['val_recall'].append(val_recall)
    history['val_f1'].append(val_f1)
    
    print(f"Epoch {epoch+1:02d}: Loss={train_loss:.4f}/{val_loss:.4f} | IoU={val_iou:.4f} | P={val_precision:.4f} | R={val_recall:.4f} | F1={val_f1:.4f}")
    
    if val_iou > best_iou:
        best_iou = val_iou
        best_epoch = epoch + 1
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'iou': best_iou,
            'history': history
        }, 'flood_model_BEST.pt')
        print(f"  ✅ New best IoU: {best_iou:.4f}")
    
    scheduler.step()

# ============================================================================
# STEP 7: Load Best Model for Testing
# ============================================================================
print("\n" + "="*70)
print("📊 TRAINING COMPLETE - LOADING BEST MODEL FOR TESTING")
print("="*70)
print(f"Best IoU: {best_iou:.4f} at Epoch {best_epoch}")

if os.path.exists('flood_model_BEST.pt'):
    checkpoint = torch.load('flood_model_BEST.pt', map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("✅ Best model loaded!")
else:
    print("⚠️ Using final model (no best checkpoint)")

# ============================================================================
# STEP 8: Generate Predictions on Test Set
# ============================================================================
print("\n🔮 Generating predictions on validation set...")

model.eval()
test_images = []
test_labels = []
test_predictions = []
test_metrics = []

with torch.no_grad():
    for images, labels in tqdm(val_loader, desc="Testing"):
        images, labels = images.to(device), labels.to(device)
        
        if torch.isnan(images).any():
            continue
        
        outputs = model(images)
        if torch.isnan(outputs).any():
            continue
        
        preds = outputs.argmax(dim=1)
        
        # Store samples for visualization
        for img, label, pred in zip(images.cpu(), labels.cpu(), preds.cpu()):
            test_images.append(img.numpy())
            test_labels.append(label.numpy())
            test_predictions.append(pred.numpy())
            
            # Calculate metrics
            tp = ((pred == 1) & (label == 1)).sum().item()
            fp = ((pred == 1) & (label == 0)).sum().item()
            fn = ((pred == 0) & (label == 1)).sum().item()
            
            iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            test_metrics.append({'iou': iou, 'precision': precision, 'recall': recall, 'f1': f1})

print(f"✅ Generated {len(test_predictions)} predictions")

# ============================================================================
# STEP 9: VISUALIZATIONS
# ============================================================================
print("\n📈 Creating visualizations...")

# Figure 1: Training History
fig = plt.figure(figsize=(20, 10))
gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

# Loss curves
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(history['train_loss'], label='Train Loss', linewidth=2)
ax1.plot(history['val_loss'], label='Val Loss', linewidth=2)
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Loss', fontsize=12)
ax1.set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# IoU over time
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(history['val_iou'], color='#2ecc71', linewidth=2, marker='o', markersize=3)
ax2.axhline(y=0.72, color='r', linestyle='--', label='Target (0.72)', linewidth=2)
ax2.axhline(y=best_iou, color='b', linestyle=':', label=f'Best ({best_iou:.4f})', linewidth=2)
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('IoU Score', fontsize=12)
ax2.set_title(f'Validation IoU (Best: {best_iou:.4f} @ Epoch {best_epoch})', fontsize=14, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

# All metrics
ax3 = fig.add_subplot(gs[0, 2])
ax3.plot(history['val_iou'], label='IoU', linewidth=2)
ax3.plot(history['val_precision'], label='Precision', linewidth=2)
ax3.plot(history['val_recall'], label='Recall', linewidth=2)
ax3.plot(history['val_f1'], label='F1 Score', linewidth=2)
ax3.set_xlabel('Epoch', fontsize=12)
ax3.set_ylabel('Score', fontsize=12)
ax3.set_title('All Metrics Over Time', fontsize=14, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)

# Metric distributions
metrics_for_plot = ['iou', 'precision', 'recall', 'f1']
for idx, metric_name in enumerate(metrics_for_plot):
    ax = fig.add_subplot(gs[1, idx % 3])
    values = [m[metric_name] for m in test_metrics]
    ax.hist(values, bins=30, edgecolor='black', alpha=0.7, color=f'C{idx}')
    ax.axvline(np.mean(values), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(values):.3f}')
    ax.set_xlabel(metric_name.upper(), fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(f'{metric_name.upper()} Distribution', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    if idx == 2:
        break

plt.suptitle('🌊 Flood Detection Model - Training & Testing Results', fontsize=18, fontweight='bold', y=0.995)
plt.savefig('01_training_results.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Saved: 01_training_results.png")

# Figure 2: Prediction Samples
fig, axes = plt.subplots(4, 6, figsize=(24, 16))
fig.suptitle('🌊 Flood Detection Predictions - Sample Results', fontsize=18, fontweight='bold')

# Select diverse samples (best, worst, median)
sorted_indices = sorted(range(len(test_metrics)), key=lambda i: test_metrics[i]['iou'])
sample_indices = [
    sorted_indices[0],  # Worst
    sorted_indices[len(sorted_indices)//4],
    sorted_indices[len(sorted_indices)//2],  # Median
    sorted_indices[3*len(sorted_indices)//4],
    sorted_indices[-1],  # Best
    sorted_indices[-2]
]

for row, idx in enumerate(sample_indices[:4]):
    # SAR Image
    axes[row, 0].imshow(test_images[idx][0], cmap='gray')
    axes[row, 0].set_title('SAR Image (VV)', fontsize=10)
    axes[row, 0].axis('off')
    
    # Ground Truth
    axes[row, 1].imshow(test_labels[idx], cmap='Blues', vmin=0, vmax=1)
    axes[row, 1].set_title('Ground Truth', fontsize=10)
    axes[row, 1].axis('off')
    
    # Prediction
    axes[row, 2].imshow(test_predictions[idx], cmap='Blues', vmin=0, vmax=1)
    axes[row, 2].set_title('Prediction', fontsize=10)
    axes[row, 2].axis('off')
    
    # Overlay (prediction on SAR)
    axes[row, 3].imshow(test_images[idx][0], cmap='gray', alpha=0.7)
    axes[row, 3].imshow(test_predictions[idx], cmap='Blues', alpha=0.5, vmin=0, vmax=1)
    axes[row, 3].set_title('Overlay', fontsize=10)
    axes[row, 3].axis('off')
    
    # Error Map (TP, FP, FN)
    error_map = np.zeros((*test_labels[idx].shape, 3))
    tp_mask = (test_predictions[idx] == 1) & (test_labels[idx] == 1)
    fp_mask = (test_predictions[idx] == 1) & (test_labels[idx] == 0)
    fn_mask = (test_predictions[idx] == 0) & (test_labels[idx] == 1)
    error_map[tp_mask] = [0, 1, 0]  # Green = True Positive
    error_map[fp_mask] = [1, 0, 0]  # Red = False Positive
    error_map[fn_mask] = [1, 1, 0]  # Yellow = False Negative
    axes[row, 4].imshow(error_map)
    axes[row, 4].set_title('Error Map', fontsize=10)
    axes[row, 4].axis('off')
    
    # Metrics
    metrics = test_metrics[idx]
    axes[row, 5].axis('off')
    metrics_text = f"""IoU:       {metrics['iou']:.4f}
Precision: {metrics['precision']:.4f}
Recall:    {metrics['recall']:.4f}
F1 Score:  {metrics['f1']:.4f}

Quality: {'⭐⭐⭐' if metrics['iou'] > 0.5 else '⭐⭐' if metrics['iou'] > 0.3 else '⭐'}"""
    axes[row, 5].text(0.1, 0.5, metrics_text, fontsize=11, verticalalignment='center', family='monospace')

plt.tight_layout()
plt.savefig('02_prediction_samples.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Saved: 02_prediction_samples.png")

# Figure 3: Summary Statistics
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('🌊 Test Set Performance Summary', fontsize=16, fontweight='bold')

# Box plots
metrics_data = {
    'IoU': [m['iou'] for m in test_metrics],
    'Precision': [m['precision'] for m in test_metrics],
    'Recall': [m['recall'] for m in test_metrics],
    'F1': [m['f1'] for m in test_metrics]
}

axes[0, 0].boxplot(metrics_data.values(), labels=metrics_data.keys())
axes[0, 0].set_ylabel('Score', fontsize=12)
axes[0, 0].set_title('Metric Distributions (Box Plot)', fontsize=14)
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].axhline(y=0.72, color='r', linestyle='--', alpha=0.5, label='Target IoU')

# Summary bar chart
means = [np.mean(v) for v in metrics_data.values()]
stds = [np.std(v) for v in metrics_data.values()]
x_pos = np.arange(len(metrics_data))
axes[0, 1].bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12'])
axes[0, 1].set_xticks(x_pos)
axes[0, 1].set_xticklabels(metrics_data.keys())
axes[0, 1].set_ylabel('Mean Score ± Std', fontsize=12)
axes[0, 1].set_title('Average Performance', fontsize=14)
axes[0, 1].grid(True, alpha=0.3, axis='y')

# Scatter: IoU vs F1
axes[1, 0].scatter([m['iou'] for m in test_metrics], [m['f1'] for m in test_metrics], alpha=0.6)
axes[1, 0].plot([0, 1], [0, 1], 'r--', alpha=0.3)
axes[1, 0].set_xlabel('IoU', fontsize=12)
axes[1, 0].set_ylabel('F1 Score', fontsize=12)
axes[1, 0].set_title('IoU vs F1 Score', fontsize=14)
axes[1, 0].grid(True, alpha=0.3)

# Final summary text
axes[1, 1].axis('off')
summary_text = f"""
📊 FINAL RESULTS
{"="*30}

Test Samples:     {len(test_metrics)}

Mean IoU:         {np.mean([m['iou'] for m in test_metrics]):.4f}
Mean Precision:   {np.mean([m['precision'] for m in test_metrics]):.4f}
Mean Recall:      {np.mean([m['recall'] for m in test_metrics]):.4f}
Mean F1:          {np.mean([m['f1'] for m in test_metrics]):.4f}

Best IoU:         {max([m['iou'] for m in test_metrics]):.4f}
Worst IoU:        {min([m['iou'] for m in test_metrics]):.4f}

Target IoU:       0.72
Status:           {'✅ ACHIEVED!' if best_iou >= 0.72 else f'📈 {(best_iou/0.72*100):.1f}% of target'}

Best Model:       Epoch {best_epoch}
Training Time:    30 epochs
Device:           {device}
"""
axes[1, 1].text(0.1, 0.5, summary_text, fontsize=11, verticalalignment='center', family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout()
plt.savefig('03_summary_statistics.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Saved: 03_summary_statistics.png")

# ============================================================================
# STEP 10: Download Everything
# ============================================================================
print("\n" + "="*70)
print("📥 DOWNLOADING RESULTS")
print("="*70)

from google.colab import files

# Download model
if os.path.exists('flood_model_BEST.pt'):
    files.download('flood_model_BEST.pt')
    print("✅ Downloaded: flood_model_BEST.pt")

# Download all plots
for img in ['01_training_results.png', '02_prediction_samples.png', '03_summary_statistics.png']:
    if os.path.exists(img):
        files.download(img)
        print(f"✅ Downloaded: {img}")

# Create final summary report
with open('RESULTS_SUMMARY.txt', 'w') as f:
    f.write("="*70 + "\n")
    f.write("🌊 FLOOD DETECTION MODEL - FINAL RESULTS\n")
    f.write("="*70 + "\n\n")
    f.write(f"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Device: {device}\n")
    f.write(f"Epochs: 30\n")
    f.write(f"Training Samples: {len(train_dataset)}\n")
    f.write(f"Validation Samples: {len(val_dataset)}\n\n")
    f.write(f"Best Validation IoU: {best_iou:.4f} (Epoch {best_epoch})\n\n")
    f.write("Test Set Performance:\n")
    f.write(f"  Mean IoU:       {np.mean([m['iou'] for m in test_metrics]):.4f}\n")
    f.write(f"  Mean Precision: {np.mean([m['precision'] for m in test_metrics]):.4f}\n")
    f.write(f"  Mean Recall:    {np.mean([m['recall'] for m in test_metrics]):.4f}\n")
    f.write(f"  Mean F1 Score:  {np.mean([m['f1'] for m in test_metrics]):.4f}\n\n")
    f.write(f"Target IoU: 0.72 - {'ACHIEVED ✅' if best_iou >= 0.72 else f'NOT MET ({best_iou/0.72*100:.1f}%) ⚠️'}\n")
    f.write("="*70 + "\n")

files.download('RESULTS_SUMMARY.txt')
print("✅ Downloaded: RESULTS_SUMMARY.txt")

print("\n" + "="*70)
print("🎉 COMPLETE! All files downloaded to your Downloads folder.")
print("="*70)
print(f"\n🏆 Final Best IoU: {best_iou:.4f}")
if best_iou >= 0.72:
    print("✅ TARGET ACHIEVED! Great job!")
elif best_iou >= 0.5:
    print("📈 Good progress! Consider training longer for better results.")
else:
    print("⚠️ Results could be better. Check data quality or try different hyperparameters.")

