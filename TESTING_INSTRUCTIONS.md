# How to Test Your Trained Flood Detection Model

## Quick Start (3 Steps)

### Step 1: Open Google Colab
1. Go to: https://colab.research.google.com/
2. Click "New notebook"
3. Change runtime to GPU (Runtime → Change runtime type → GPU)

### Step 2: Copy Test Script
1. Open `TEST_MODEL_COLAB.py` in this project
2. Copy ALL the code (Ctrl+A, Ctrl+C)
3. Paste into the Colab cell
4. Click Run (▶️)

### Step 3: Upload Your Model
When prompted:
1. Upload your trained model (`flood_model_best.pt`)
2. Wait for testing to complete (~5 minutes)
3. Download the result images

---

## What the Test Will Show

### 📊 Metrics:
- **IoU (Intersection over Union)** - Target: > 0.72
- **Precision** - How accurate are flood predictions
- **Recall** - How much of the flood was detected
- **F1 Score** - Overall performance

### 📸 Visualizations:
1. **Side-by-side comparison**:
   - Original SAR image
   - Ground truth flood mask
   - Model prediction

2. **Metrics distribution**:
   - Histogram of IoU scores
   - Precision/Recall distribution
   - F1 score across test images

---

## Expected Results

### ✅ Good Performance:
- IoU > 0.70
- F1 Score > 0.75
- Clear flood boundaries in predictions

### ⚠️ Needs Improvement:
- IoU < 0.60
- Many false positives/negatives
- Blurry flood boundaries

**If performance is low:**
- Train for more epochs
- Use ensemble of models
- Add more training data

---

## After Testing

### If Results Are Good:
1. ✅ Save your model
2. Train additional models (Swin, MaxViT)
3. Create ensemble for better accuracy
4. Deploy to cloud

### If Results Need Improvement:
1. Train longer (more epochs)
2. Try different model architectures
3. Adjust hyperparameters
4. Add data augmentation

---

## Example Output

```
📊 TEST RESULTS
============================================================
IoU            : 0.7245
Precision      : 0.8123
Recall         : 0.7834
F1 Score       : 0.7976
============================================================
🎉 ✅ TARGET MET! IoU > 0.72
```

---

## Troubleshooting

**Issue**: Model won't load  
**Fix**: Make sure you uploaded the correct `.pt` file

**Issue**: Low accuracy  
**Fix**: Train for more epochs or use better architecture

**Issue**: Out of memory  
**Fix**: Reduce batch size or image resolution

---

## Next Steps After Testing

1. ✅ **Document your results** - Save metrics and images
2. 🚀 **Train more models** - Swin, MaxViT for ensemble
3. 🌍 **Test on real floods** - Download fresh satellite data
4. 📱 **Deploy** - Share your working model

Ready to test? Open Colab and run the script!

