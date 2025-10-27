# ❌ Why Your Model Has 0.0000 Performance

## The Problem

Your model is predicting **ALL pixels as class 0 (background)** and **ZERO pixels as class 1 (flood)**. That's why all metrics are 0.0000.

---

## Root Causes

### 1. **Extreme Class Imbalance** ⚖️
Flood pixels are **very rare** in satellite images (typically 2-10% of total pixels). During training, the model learned:
- "If I always predict NO FLOOD, I'm right 90-98% of the time!"
- This gives low loss but terrible flood detection

### 2. **No Class Weights in Loss Function** 📉
Your training script uses standard `CrossEntropyLoss` which treats all pixels equally:
```python
criterion = nn.CrossEntropyLoss()  # ❌ Treats rare flood pixels same as common background
```

The model optimizes for overall accuracy, so it learns to predict background everywhere.

### 3. **Training Data Issues** 📊
Possible issues:
- Labels might have very few actual flood pixels
- TIFF files weren't read properly during training (PIL vs rasterio)
- Normalization mismatch between training and testing

---

## The Solution

### ✅ Use Weighted Loss Function

Add class weights to make the model care about flood pixels:

```python
# Calculate weights based on class frequency
weight_background = 1.0
weight_flood = 10.0  # Or higher based on actual imbalance

class_weights = torch.tensor([weight_background, weight_flood]).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
```

This tells the model: **"Missing a flood pixel costs 10x more than missing a background pixel"**

### ✅ Use Focal Loss

Focal Loss focuses on hard-to-classify pixels (like floods):

```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()
```

### ✅ Monitor IoU During Training

Don't just track loss - track **IoU** to see if the model is actually detecting floods:

```python
# After each epoch
val_iou = calculate_iou(predictions, labels)
print(f"Validation IoU: {val_iou:.4f}")
```

---

## Quick Fix: Run the Diagnostic Script

I created `DIAGNOSE_AND_FIX_MODEL.py` which will:

1. ✅ Check if your labels actually contain flood pixels
2. ✅ Analyze what your model is predicting
3. ✅ Calculate proper class weights from your data
4. ✅ **Automatically retrain** with weighted loss
5. ✅ Download the fixed model

### How to Use:

1. **Open Google Colab**
2. **Copy `DIAGNOSE_AND_FIX_MODEL.py`**
3. **Paste into a Colab cell**
4. **Run it!**

It will:
- Diagnose the exact problem
- Ask if you want to retrain automatically
- If yes: retrain with proper class weighting
- Download the fixed model: `flood_model_FIXED.pt`

---

## What to Expect After Fix

### Before (Current):
```
IoU:       0.0000  ❌
Precision: 0.0000  ❌
Recall:    0.0000  ❌
F1 Score:  0.0000  ❌
```

### After (With Weighted Loss):
```
IoU:       0.30-0.60  ✅ (depending on data quality)
Precision: 0.40-0.70  ✅
Recall:    0.50-0.80  ✅
F1 Score:  0.40-0.70  ✅
```

**Target: IoU > 0.72** requires:
- More training data (you have small dataset)
- Longer training (50-100 epochs)
- Better model architecture (ResNet-50 UNet, not simplified version)

---

## Manual Fix (If You Want to Retrain Yourself)

Update your training script:

### 1. Add Class Weights
```python
# Before training loop, calculate weights
flood_ratio = 0.05  # Adjust based on your data
weight_flood = (1 - flood_ratio) / flood_ratio  # ~19 if 5% flood pixels

class_weights = torch.tensor([1.0, weight_flood]).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
```

### 2. Monitor IoU
```python
def calculate_iou(pred, target):
    intersection = ((pred == 1) & (target == 1)).sum()
    union = ((pred == 1) | (target == 1)).sum()
    return (intersection / union).item() if union > 0 else 0
```

### 3. Train Longer
```python
num_epochs = 50  # Instead of 10
```

### 4. Use Lower Learning Rate
```python
optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Instead of 0.001
```

---

## Why This Happens in Flood Detection

Flood detection is a **highly imbalanced segmentation problem**:

| Pixel Type | % of Image | Model Behavior Without Weighting |
|------------|------------|----------------------------------|
| Background | 90-98% | ✅ Learns well (common) |
| Flood | 2-10% | ❌ Ignores (rare) |

Standard loss functions optimize for **overall accuracy**, so the model learns:
- "Predict all background" → 95% accuracy ✅ (but useless)
- "Detect floods" → 75% accuracy ❌ (but useful!)

**Solution**: Weight the loss to make flood pixels more important!

---

## Next Steps

### Option 1: Automated Fix (Recommended) ⚡
```bash
1. Run DIAGNOSE_AND_FIX_MODEL.py in Colab
2. Let it retrain automatically (~15 min)
3. Download flood_model_FIXED.pt
4. Test again
```

### Option 2: Manual Fix 🛠️
```bash
1. Update training script with class weights
2. Retrain for 50 epochs
3. Monitor IoU during training
4. Save best model based on IoU (not loss)
```

### Option 3: Better Architecture 🚀
```bash
1. Use pre-trained ResNet-50 UNet (from original plan)
2. Use larger dataset (full Sen1Floods11, not just 1-2GB)
3. Train ensemble of models
4. Use Focal Loss + Dice Loss combination
```

---

## Expected Timeline

| Approach | Time | Expected IoU |
|----------|------|--------------|
| Quick fix with weights | 15 min | 0.30-0.50 |
| Proper retraining (50 epochs) | 1 hour | 0.50-0.65 |
| Full architecture + data | 4-6 hours | 0.72+ (target) |

---

**Ready to fix it?** Run `DIAGNOSE_AND_FIX_MODEL.py` now! 🚀

