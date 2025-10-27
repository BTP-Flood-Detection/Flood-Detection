"""
SINGLE-CELL COLAB TRAINING SCRIPT
Copy this entire file and paste into a Google Colab cell, then run it.

Prerequisites:
1. Upload your data to Google Drive at: My Drive/Flood detection/data/
   - Should have: data/raw/S1/ and data/raw/labels/
2. Change runtime to GPU (Runtime → Change runtime type → GPU)
3. Run this cell
"""

# ============================================================================
# STEP 1: Install Dependencies
# ============================================================================
print("📦 Installing dependencies...")
import subprocess
import sys

packages = [
    "torch", "torchvision", "timm", "albumentations", 
    "segmentation-models-pytorch", "opencv-python-headless",
    "pyyaml", "scikit-learn", "tqdm"
]

for package in packages:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])

print("✅ Dependencies installed!\n")

# ============================================================================
# STEP 2: Mount Google Drive
# ============================================================================
print("📂 Mounting Google Drive...")
from google.colab import drive
drive.mount('/content/drive')
print("✅ Drive mounted!\n")

# ============================================================================
# STEP 3: Setup Data Paths
# ============================================================================
import os
import shutil

# Copy data from Drive to Colab (faster training)
print("📥 Copying data from Google Drive...")
src_data = '/content/drive/MyDrive/Flood detection/data'
dst_data = '/content/flood_detection/data'

if os.path.exists(src_data):
    shutil.copytree(src_data, dst_data)
    print(f"✅ Data copied to {dst_data}")
else:
    print(f"❌ ERROR: Data not found at {src_data}")
    print("Please upload your data to Google Drive at: My Drive/Flood detection/data/")
    raise FileNotFoundError("Data not found in Google Drive")

# Check what we have
s1_files = len(os.listdir(f"{dst_data}/raw/S1"))
label_files = len(os.listdir(f"{dst_data}/raw/labels"))
print(f"\n📊 Dataset: {s1_files} S1 images, {label_files} labels")

# ============================================================================
# STEP 4: Create Training Code Inline
# ============================================================================
print("\n⚙️ Setting up training code...")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torchvision.models import resnet50, ResNet50_Weights
import torch.nn.functional as F

# Check device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️ Using device: {device}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")

# ============================================================================
# Dataset Class
# ============================================================================
class FloodDataset(Dataset):
    def __init__(self, s1_paths, label_paths, augment=False):
        self.s1_paths = s1_paths
        self.label_paths = label_paths
        self.augment = augment
    
    def __len__(self):
        return len(self.s1_paths)
    
    def __getitem__(self, idx):
        # Load S1 image (using PIL for simplicity)
        from PIL import Image
        import torchvision.transforms as T
        
        # Load image (assuming single-band TIFF)
        try:
            s1_img = Image.open(self.s1_paths[idx])
            s1_array = np.array(s1_img, dtype=np.float32)
            
            # If multi-band, take first 2 channels, else replicate
            if len(s1_array.shape) == 3:
                s1_array = s1_array[:, :, :2]
            else:
                s1_array = np.stack([s1_array, s1_array], axis=-1)
            
            # Load label
            label_img = Image.open(self.label_paths[idx])
            label_array = np.array(label_img, dtype=np.float32)
            if len(label_array.shape) == 3:
                label_array = label_array[:, :, 0]
            
            # Normalize to 0-1
            label_array = (label_array > 0).astype(np.float32)
            
            # Resize to consistent size
            target_size = (256, 256)
            s1_tensor = T.functional.to_tensor(s1_array)
            s1_tensor = T.functional.resize(s1_tensor, target_size)
            
            label_tensor = torch.from_numpy(label_array).unsqueeze(0)
            label_tensor = T.functional.resize(label_tensor, target_size)
            label_tensor = label_tensor.squeeze(0).long()
            
            # Normalize S1
            s1_tensor = (s1_tensor - s1_tensor.mean()) / (s1_tensor.std() + 1e-8)
            
            return s1_tensor, label_tensor
            
        except Exception as e:
            print(f"Error loading {self.s1_paths[idx]}: {e}")
            # Return dummy data
            return torch.zeros(2, 256, 256), torch.zeros(256, 256).long()

# ============================================================================
# Model Architecture (Simplified ResNet UNet)
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
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        
        e3 = self.enc3(p2)
        
        # Decoder
        u1 = self.up1(e3)
        d1 = torch.cat([u1, e2], dim=1)
        d1 = self.dec1(d1)
        
        u2 = self.up2(d1)
        d2 = torch.cat([u2, e1], dim=1)
        d2 = self.dec2(d2)
        
        out = self.final(d2)
        return out

# ============================================================================
# Metrics
# ============================================================================
def calculate_iou(pred, target, num_classes=2):
    ious = []
    pred = pred.argmax(dim=1)
    
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)
        
        intersection = (pred_cls & target_cls).float().sum()
        union = (pred_cls | target_cls).float().sum()
        
        if union > 0:
            iou = intersection / union
        else:
            iou = 1.0 if intersection == 0 else 0.0
        
        ious.append(iou.item())
    
    return np.mean(ious)

# ============================================================================
# STEP 5: Prepare Data
# ============================================================================
print("\n📊 Preparing dataset...")

s1_dir = Path(f"{dst_data}/raw/S1")
labels_dir = Path(f"{dst_data}/raw/labels")

s1_files = sorted(list(s1_dir.glob("*.tif")))[:100]  # Use first 100 for speed
label_files = sorted(list(labels_dir.glob("*.tif")))[:100]

print(f"Found {len(s1_files)} S1 images and {len(label_files)} labels")

# Split data
n_train = int(0.7 * len(s1_files))
n_val = int(0.15 * len(s1_files))

train_s1 = [str(f) for f in s1_files[:n_train]]
train_labels = [str(f) for f in label_files[:n_train]]

val_s1 = [str(f) for f in s1_files[n_train:n_train+n_val]]
val_labels = [str(f) for f in label_files[n_train:n_train+n_val]]

print(f"Train: {len(train_s1)}, Val: {len(val_s1)}")

# Create datasets
train_dataset = FloodDataset(train_s1, train_labels, augment=True)
val_dataset = FloodDataset(val_s1, val_labels, augment=False)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=2)

# ============================================================================
# STEP 6: Train Model
# ============================================================================
print("\n🚀 Starting training...")

model = SimpleFloodModel(input_channels=2, num_classes=2).to(device)
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
criterion = nn.CrossEntropyLoss()

best_iou = 0.0
epochs = 20

for epoch in range(epochs):
    # Train
    model.train()
    train_loss = 0.0
    
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    # Validate
    model.eval()
    val_ious = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            iou = calculate_iou(outputs, labels)
            val_ious.append(iou)
    
    mean_iou = np.mean(val_ious)
    
    print(f"Epoch {epoch+1}: Loss={train_loss/len(train_loader):.4f}, IoU={mean_iou:.4f}")
    
    # Save best model
    if mean_iou > best_iou:
        best_iou = mean_iou
        torch.save({
            'model_state_dict': model.state_dict(),
            'iou': best_iou,
            'epoch': epoch
        }, '/content/flood_model_best.pt')
        print(f"   ✅ New best model saved! IoU: {best_iou:.4f}")

# ============================================================================
# STEP 7: Save and Download Model
# ============================================================================
print(f"\n✅ Training complete!")
print(f"Best IoU: {best_iou:.4f}")

# Download model
from google.colab import files
files.download('/content/flood_model_best.pt')

print("\n📥 Model downloaded!")
print("Copy to: C:\\Users\\Nishant Raj\\Desktop\\Flood detection\\models\\resnet_best.pt")
print("\nThen run: streamlit run app/streamlit_app.py")

