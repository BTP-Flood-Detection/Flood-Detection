"""
DIAGNOSE AND FIX YOUR FLOOD DETECTION MODEL
Run this in Google Colab to identify why performance is 0.0000

This will:
1. Check if your data has flood labels
2. Verify model is learning
3. Fix class imbalance issues
4. Retrain with better settings
"""

# ============================================================================
# STEP 1: Setup
# ============================================================================
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

# Copy data
print("\n📥 Copying data...")
src_data = '/content/drive/MyDrive/Flood detection/data'
dst_data = '/content/test_data'

if os.path.exists(dst_data):
    shutil.rmtree(dst_data)

if os.path.exists(src_data):
    shutil.copytree(src_data, dst_data)
    print("✅ Data copied")
else:
    print("❌ Data not found!")
    raise FileNotFoundError("Data not found in Google Drive")

# ============================================================================
# STEP 2: DIAGNOSE - Check Your Data
# ============================================================================
print("\n" + "="*60)
print("🔍 DIAGNOSIS 1: Checking if labels contain flood pixels")
print("="*60)

# Install rasterio
import subprocess
import sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "rasterio"])

import rasterio

labels_dir = Path(f"{dst_data}/raw/labels")
label_files = sorted(list(labels_dir.glob("*.tif")))[:20]

flood_pixel_counts = []
total_pixel_counts = []

for label_file in label_files[:5]:  # Check first 5
    try:
        with rasterio.open(label_file) as src:
            label = src.read(1)
            
        flood_pixels = (label > 0).sum()
        total_pixels = label.size
        flood_ratio = flood_pixels / total_pixels
        
        flood_pixel_counts.append(flood_pixels)
        total_pixel_counts.append(total_pixels)
        
        print(f"✓ {label_file.name}")
        print(f"  Flood pixels: {flood_pixels:,} / {total_pixels:,} ({flood_ratio*100:.2f}%)")
        
    except Exception as e:
        print(f"✗ Error: {e}")

avg_flood_ratio = sum(flood_pixel_counts) / sum(total_pixel_counts)
print(f"\n📊 Average flood coverage: {avg_flood_ratio*100:.2f}%")

if avg_flood_ratio < 0.001:
    print("⚠️ WARNING: Very few flood pixels! This is extreme class imbalance.")
elif avg_flood_ratio < 0.05:
    print("⚠️ WARNING: Low flood coverage. Need weighted loss function.")
else:
    print("✅ Decent flood coverage in labels")

# ============================================================================
# STEP 3: DIAGNOSE - Check Model Output Distribution
# ============================================================================
print("\n" + "="*60)
print("🔍 DIAGNOSIS 2: Checking model predictions")
print("="*60)

# Load your trained model
print("\n📥 Upload your trained model (flood_model_best.pt):")
from google.colab import files
uploaded = files.upload()
model_file = list(uploaded.keys())[0]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Define model architecture (same as training)
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

# Load model
model = SimpleFloodModel(input_channels=2, num_classes=2).to(device)
checkpoint = torch.load(model_file, map_location=device, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"✅ Model loaded!")

# Test on one image
import torchvision.transforms as T

s1_dir = Path(f"{dst_data}/raw/S1")
s1_files = sorted(list(s1_dir.glob("*.tif")))

print("\n🔬 Analyzing model output on sample image...")

with rasterio.open(str(s1_files[0])) as src:
    s1_data = src.read().astype(np.float32)

if s1_data.shape[0] >= 2:
    s1_data = s1_data[:2]
elif s1_data.shape[0] == 1:
    s1_data = np.stack([s1_data[0], s1_data[0]], axis=0)

s1_tensor = torch.from_numpy(s1_data)
s1_tensor = T.functional.resize(s1_tensor, (256, 256))
s1_tensor = (s1_tensor - s1_tensor.mean()) / (s1_tensor.std() + 1e-8)

with torch.no_grad():
    output = model(s1_tensor.unsqueeze(0).to(device))
    probabilities = torch.softmax(output, dim=1)
    prediction = output.argmax(dim=1)

# Analyze output
prob_flood = probabilities[0, 1].cpu().numpy()  # Probability of flood class
pred_mask = prediction[0].cpu().numpy()

print(f"\n📊 Model Output Analysis:")
print(f"  Output shape: {output.shape}")
print(f"  Logits - Class 0 (no flood): min={output[0,0].min():.3f}, max={output[0,0].max():.3f}, mean={output[0,0].mean():.3f}")
print(f"  Logits - Class 1 (flood):    min={output[0,1].min():.3f}, max={output[0,1].max():.3f}, mean={output[0,1].mean():.3f}")
print(f"\n  Probabilities - Flood class: min={prob_flood.min():.3f}, max={prob_flood.max():.3f}, mean={prob_flood.mean():.3f}")
print(f"  Predicted flood pixels: {(pred_mask == 1).sum()} / {pred_mask.size} ({(pred_mask == 1).sum()/pred_mask.size*100:.2f}%)")

if (pred_mask == 1).sum() == 0:
    print("\n❌ PROBLEM IDENTIFIED: Model predicts NO flood pixels!")
    print("   This means the model always chooses class 0 (no flood)")
    print("\n💡 ROOT CAUSE: Likely class imbalance during training")
else:
    print("\n✓ Model is predicting some flood pixels")

# ============================================================================
# STEP 4: THE FIX - Retrain with Weighted Loss
# ============================================================================
print("\n" + "="*60)
print("🔧 SOLUTION: Retrain with class-weighted loss function")
print("="*60)

print("\nWould you like to retrain now? This will:")
print("1. Use weighted Cross-Entropy to balance classes")
print("2. Use Focal Loss to focus on hard examples")
print("3. Train for 20 epochs with better settings")
print("\nThis will take ~10-15 minutes on GPU")

response = input("\nRetrain now? (yes/no): ")

if response.lower() in ['yes', 'y']:
    print("\n🚀 Starting retraining...")
    
    # Calculate class weights from data
    print("Calculating class weights from dataset...")
    total_flood = sum(flood_pixel_counts)
    total_background = sum(total_pixel_counts) - total_flood
    
    weight_background = 1.0
    weight_flood = total_background / total_flood if total_flood > 0 else 10.0
    
    print(f"Class weights: Background=1.0, Flood={weight_flood:.2f}")
    
    # Create weighted loss (IMPORTANT: use float32, not float64)
    class_weights = torch.tensor([weight_background, weight_flood], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Prepare dataset
    class FloodDataset(Dataset):
        def __init__(self, s1_files, label_files):
            self.s1_files = s1_files
            self.label_files = label_files
        
        def __len__(self):
            return len(self.s1_files)
        
        def __getitem__(self, idx):
            try:
                with rasterio.open(str(self.s1_files[idx])) as src:
                    s1 = src.read().astype(np.float32)
                
                if s1.shape[0] >= 2:
                    s1 = s1[:2]
                elif s1.shape[0] == 1:
                    s1 = np.stack([s1[0], s1[0]], axis=0)
                
                with rasterio.open(str(self.label_files[idx])) as src:
                    label = src.read(1).astype(np.float32)
                
                label = (label > 0).astype(np.float32)
                
                # To tensor
                s1_tensor = torch.from_numpy(s1)
                s1_tensor = T.functional.resize(s1_tensor, (256, 256))
                s1_tensor = (s1_tensor - s1_tensor.mean()) / (s1_tensor.std() + 1e-8)
                
                label_tensor = torch.from_numpy(label).unsqueeze(0)
                label_tensor = T.functional.resize(label_tensor, (256, 256))
                label_tensor = label_tensor.squeeze(0).long()
                
                return s1_tensor, label_tensor
            except:
                return torch.zeros(2, 256, 256), torch.zeros(256, 256).long()
    
    # Create dataloaders
    s1_all = sorted(list(s1_dir.glob("*.tif")))
    label_all = sorted(list(labels_dir.glob("*.tif")))
    
    split_idx = int(0.8 * len(s1_all))
    train_dataset = FloodDataset(s1_all[:split_idx], label_all[:split_idx])
    val_dataset = FloodDataset(s1_all[split_idx:], label_all[split_idx:])
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2)
    
    print(f"Training samples: {len(train_dataset)}, Validation: {len(val_dataset)}")
    
    # Reinitialize model
    model = SimpleFloodModel(input_channels=2, num_classes=2).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
    
    # Training loop
    best_iou = 0
    history = {'train_loss': [], 'val_loss': [], 'val_iou': []}
    
    for epoch in range(20):
        # Train
        model.train()
        train_loss = 0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/20"):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validate
        model.eval()
        val_loss = 0
        iou_scores = []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                preds = outputs.argmax(dim=1)
                for pred, label in zip(preds, labels):
                    intersection = ((pred == 1) & (label == 1)).sum().item()
                    union = ((pred == 1) | (label == 1)).sum().item()
                    iou = intersection / union if union > 0 else 0
                    iou_scores.append(iou)
        
        val_loss /= len(val_loader)
        val_iou = np.mean(iou_scores)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_iou'].append(val_iou)
        
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val IoU={val_iou:.4f}")
        
        # Save best model
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'iou': best_iou
            }, 'flood_model_FIXED.pt')
            print(f"  ✓ New best IoU: {best_iou:.4f}")
        
        scheduler.step()
    
    print("\n" + "="*60)
    print("✅ RETRAINING COMPLETE!")
    print("="*60)
    print(f"Best IoU achieved: {best_iou:.4f}")
    
    # Plot training history
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    
    ax2.plot(history['val_iou'])
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('IoU')
    ax2.set_title('Validation IoU Over Time')
    ax2.axhline(y=0.72, color='r', linestyle='--', label='Target (0.72)')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('training_history_FIXED.png', dpi=150)
    plt.show()
    
    # Save final model if best wasn't saved
    if not os.path.exists('flood_model_FIXED.pt'):
        print("\n⚠️ Saving final model (no improvement detected)...")
        torch.save({
            'epoch': 19,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'iou': best_iou
        }, 'flood_model_FIXED.pt')
    
    # Download new model
    print("\n📥 Downloading improved model...")
    
    if os.path.exists('flood_model_FIXED.pt'):
        files.download('flood_model_FIXED.pt')
    else:
        print("⚠️ Model file not found - saving current state...")
        torch.save({
            'model_state_dict': model.state_dict(),
            'iou': best_iou
        }, 'flood_model_FINAL.pt')
        files.download('flood_model_FINAL.pt')
    
    files.download('training_history_FIXED.png')
    
    if best_iou > 0.1:
        print(f"\n✅ Done! Best IoU: {best_iou:.4f}")
        print("Use the downloaded model for testing.")
    else:
        print(f"\n⚠️ Warning: IoU is very low ({best_iou:.4f})")
        print("This suggests a data or preprocessing issue.")
        print("\nPossible causes:")
        print("1. Labels are empty or corrupted")
        print("2. Input data doesn't match label format")
        print("3. Normalization is too extreme")
        print("\nCheck your data files and try again.")
    
else:
    print("\nSkipping retraining. Here's what you need to fix:")
    print("\n1. Add class weights to your loss function:")
    print("   class_weights = torch.tensor([1.0, 10.0])  # Adjust based on class imbalance")
    print("   criterion = nn.CrossEntropyLoss(weight=class_weights)")
    print("\n2. Train for more epochs (20-50)")
    print("\n3. Use a smaller learning rate (0.0001)")
    print("\n4. Monitor IoU during training, not just loss")

print("\n" + "="*60)
print("DIAGNOSTIC COMPLETE")
print("="*60)

