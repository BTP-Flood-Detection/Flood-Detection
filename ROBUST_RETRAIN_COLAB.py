"""
ROBUST FLOOD DETECTION RETRAINING
Handles NaN values and data corruption issues
"""

print("📂 Mounting Google Drive...")
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

# Install rasterio
import subprocess
import sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "rasterio"])

import rasterio
import torchvision.transforms as T

# Copy data
print("\n📥 Copying data...")
src_data = '/content/drive/MyDrive/Flood detection/data'
dst_data = '/content/test_data'

if os.path.exists(dst_data):
    shutil.rmtree(dst_data)

shutil.copytree(src_data, dst_data)
print("✅ Data copied")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n🖥️ Using device: {device}")

# ============================================================================
# ROBUST Dataset with NaN/Inf checks
# ============================================================================
class RobustFloodDataset(Dataset):
    def __init__(self, s1_files, label_files):
        self.s1_files = s1_files
        self.label_files = label_files
        
        # Pre-validate files
        print("🔍 Validating data files...")
        self.valid_indices = []
        for idx in tqdm(range(len(s1_files))):
            if self._validate_sample(idx):
                self.valid_indices.append(idx)
        
        print(f"✅ Valid samples: {len(self.valid_indices)} / {len(s1_files)}")
    
    def _validate_sample(self, idx):
        """Check if a sample is valid (no NaN/Inf)"""
        try:
            with rasterio.open(str(self.s1_files[idx])) as src:
                s1 = src.read().astype(np.float32)
            
            with rasterio.open(str(self.label_files[idx])) as src:
                label = src.read(1).astype(np.float32)
            
            # Check for NaN or Inf
            if np.isnan(s1).any() or np.isinf(s1).any():
                return False
            if np.isnan(label).any() or np.isinf(label).any():
                return False
            
            # Check if completely zero
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
            # Load S1
            with rasterio.open(str(self.s1_files[real_idx])) as src:
                s1 = src.read().astype(np.float32)
            
            # Handle channels
            if s1.shape[0] >= 2:
                s1 = s1[:2]
            elif s1.shape[0] == 1:
                s1 = np.stack([s1[0], s1[0]], axis=0)
            else:
                raise ValueError(f"Unexpected S1 shape: {s1.shape}")
            
            # Load label
            with rasterio.open(str(self.label_files[real_idx])) as src:
                label = src.read(1).astype(np.float32)
            
            # Binary label
            label = (label > 0).astype(np.float32)
            
            # Convert to tensors
            s1_tensor = torch.from_numpy(s1)
            s1_tensor = T.functional.resize(s1_tensor, (256, 256))
            
            label_tensor = torch.from_numpy(label).unsqueeze(0)
            label_tensor = T.functional.resize(label_tensor, (256, 256))
            label_tensor = label_tensor.squeeze(0).long()
            
            # ROBUST NORMALIZATION with clipping
            # Clip extreme values first
            s1_tensor = torch.clamp(s1_tensor, -100, 100)
            
            # Normalize per-channel
            for c in range(s1_tensor.shape[0]):
                channel = s1_tensor[c]
                mean = channel.mean()
                std = channel.std()
                
                # Avoid division by zero
                if std < 1e-6:
                    std = 1.0
                
                s1_tensor[c] = (channel - mean) / std
            
            # Final safety check
            s1_tensor = torch.clamp(s1_tensor, -10, 10)
            
            # Verify no NaN
            if torch.isnan(s1_tensor).any() or torch.isinf(s1_tensor).any():
                print(f"⚠️ NaN detected in sample {real_idx}, replacing with zeros")
                s1_tensor = torch.zeros_like(s1_tensor)
            
            return s1_tensor, label_tensor
            
        except Exception as e:
            print(f"Error loading {real_idx}: {e}")
            # Return zero tensor as fallback
            return torch.zeros(2, 256, 256), torch.zeros(256, 256).long()

# ============================================================================
# Model Architecture
# ============================================================================
class SimpleFloodModel(nn.Module):
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
# Prepare Data
# ============================================================================
print("\n📊 Preparing dataset...")
s1_dir = Path(f"{dst_data}/raw/S1")
labels_dir = Path(f"{dst_data}/raw/labels")

s1_all = sorted(list(s1_dir.glob("*.tif")))
label_all = sorted(list(labels_dir.glob("*.tif")))

print(f"Found {len(s1_all)} S1 images and {len(label_all)} labels")

# Split
split_idx = int(0.8 * len(s1_all))
train_dataset = RobustFloodDataset(s1_all[:split_idx], label_all[:split_idx])
val_dataset = RobustFloodDataset(s1_all[split_idx:], label_all[split_idx:])

if len(train_dataset) == 0 or len(val_dataset) == 0:
    print("❌ ERROR: No valid training data!")
    print("Your data files appear to be corrupted or in an incompatible format.")
    raise RuntimeError("No valid data")

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2)

print(f"✅ Training samples: {len(train_dataset)}, Validation: {len(val_dataset)}")

# ============================================================================
# Setup Training
# ============================================================================
print("\n⚙️ Setting up training...")

model = SimpleFloodModel(input_channels=2, num_classes=2).to(device)
optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.01)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

# Weighted loss
class_weights = torch.tensor([1.0, 5.0], dtype=torch.float32).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)

print("✅ Model initialized")

# ============================================================================
# Training Loop with NaN detection
# ============================================================================
print("\n🚀 Starting training...")

best_iou = 0
best_epoch = 0
history = {'train_loss': [], 'val_loss': [], 'val_iou': []}

for epoch in range(30):
    # Train
    model.train()
    train_loss = 0
    nan_count = 0
    
    for batch_idx, (images, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/30")):
        images, labels = images.to(device), labels.to(device)
        
        # Check for NaN in batch
        if torch.isnan(images).any():
            nan_count += 1
            continue
        
        optimizer.zero_grad()
        outputs = model(images)
        
        # Check for NaN in outputs
        if torch.isnan(outputs).any() or torch.isinf(outputs).any():
            print(f"⚠️ NaN in model output at batch {batch_idx}, skipping...")
            nan_count += 1
            continue
        
        loss = criterion(outputs, labels)
        
        # Check for NaN in loss
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"⚠️ NaN loss at batch {batch_idx}, skipping...")
            nan_count += 1
            continue
        
        loss.backward()
        
        # Gradient clipping to prevent explosion
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        train_loss += loss.item()
    
    train_loss /= max(len(train_loader) - nan_count, 1)
    
    # Validate
    model.eval()
    val_loss = 0
    iou_scores = []
    
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
            for pred, label in zip(preds, labels):
                intersection = ((pred == 1) & (label == 1)).sum().item()
                union = ((pred == 1) | (label == 1)).sum().item()
                iou = intersection / union if union > 0 else 0
                iou_scores.append(iou)
    
    val_loss /= max(len(val_loader), 1)
    val_iou = np.mean(iou_scores) if iou_scores else 0
    
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['val_iou'].append(val_iou)
    
    print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val IoU={val_iou:.4f}")
    if nan_count > 0:
        print(f"  ⚠️ Skipped {nan_count} batches due to NaN")
    
    # Save best model
    if val_iou > best_iou:
        best_iou = val_iou
        best_epoch = epoch + 1
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'iou': best_iou
        }, 'flood_model_ROBUST.pt')
        print(f"  ✅ New best IoU: {best_iou:.4f}")
    
    scheduler.step()

# ============================================================================
# Results
# ============================================================================
print("\n" + "="*60)
print("✅ TRAINING COMPLETE!")
print("="*60)
print(f"Best IoU: {best_iou:.4f} (Epoch {best_epoch})")

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(history['train_loss'], label='Train Loss')
ax1.plot(history['val_loss'], label='Val Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training Progress')
ax1.legend()
ax1.grid(True)

ax2.plot(history['val_iou'])
ax2.set_xlabel('Epoch')
ax2.set_ylabel('IoU')
ax2.set_title(f'Validation IoU (Best: {best_iou:.4f})')
ax2.axhline(y=0.72, color='r', linestyle='--', label='Target')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig('training_ROBUST.png', dpi=150)
plt.show()

# Download
from google.colab import files

if os.path.exists('flood_model_ROBUST.pt'):
    print("\n📥 Downloading model...")
    files.download('flood_model_ROBUST.pt')
    files.download('training_ROBUST.png')
    print("✅ Done!")
else:
    print("⚠️ No model saved (IoU never improved)")

if best_iou > 0.1:
    print(f"\n🎉 Success! IoU: {best_iou:.4f}")
else:
    print(f"\n⚠️ IoU still very low: {best_iou:.4f}")
    print("This suggests fundamental data issues.")
    print("Consider using a different dataset or checking data format.")

