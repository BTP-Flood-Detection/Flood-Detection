"""
Simplified Streamlit App - No PyTorch Required
For viewing results and uploading data
"""

import streamlit as st
import numpy as np
from pathlib import Path
import os

# Page configuration
st.set_page_config(
    page_title="Flood Detection System",
    page_icon="🌊",
    layout="wide"
)

# Title
st.title("🌊 Flood Detection System")
st.markdown("### Status Dashboard")

# Show project status
st.info("""
**Current Status:**
- ✅ Complete codebase implemented
- ✅ Dataset downloaded (446 images)
- ✅ Model trained on Google Colab
- ⚠️ PyTorch DLL issue on Windows (use Colab for predictions)
""")

# Project structure
st.header("📁 Project Structure")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Code Files")
    st.code("""
✅ src/data/
   - dataset.py
   - preprocessing.py
   - gee_downloader.py
   
✅ src/models/
   - resnet_unet.py
   - swin_transformer.py
   - max_vit.py
   - ensemble.py
   
✅ src/training/
   - train.py
   - losses.py
   - metrics.py
   
✅ src/inference/
   - predict.py
   - uncertainty.py
    """)

with col2:
    st.subheader("Data & Models")
    
    # Check data
    data_path = Path("data/raw")
    if data_path.exists():
        s1_files = list((data_path / "S1").glob("*.tif")) if (data_path / "S1").exists() else []
        label_files = list((data_path / "labels").glob("*.tif")) if (data_path / "labels").exists() else []
        
        st.metric("S1 Images", len(s1_files))
        st.metric("Labels", len(label_files))
    else:
        st.warning("Data directory not found")
    
    # Check models
    models_path = Path("models")
    if models_path.exists():
        model_files = list(models_path.glob("*.pt"))
        st.metric("Trained Models", len(model_files))
        if model_files:
            for model in model_files:
                st.success(f"✅ {model.name}")
    else:
        st.warning("No trained models found")

# Usage instructions
st.header("🚀 How to Use")

tab1, tab2, tab3 = st.tabs(["Training", "Prediction", "Deployment"])

with tab1:
    st.markdown("""
    ### Training Models (Use Google Colab)
    
    **Steps:**
    1. Open Google Colab: https://colab.research.google.com/
    2. Upload `COLAB_TRAINING_SINGLE_CELL.py`
    3. Copy all code into a cell
    4. Run the cell
    5. Download trained model
    6. Copy to `models/` folder
    
    **Status:**
    - Training script: ✅ Ready
    - Dataset: ✅ Available (446 samples)
    - Colab notebook: ✅ Created
    """)

with tab2:
    st.markdown("""
    ### Making Predictions (Use Google Colab)
    
    **Option 1: Colab Inference**
    ```python
    # In Colab, after training:
    from PIL import Image
    import torch
    
    # Load model
    model.load_state_dict(torch.load('flood_model_best.pt'))
    
    # Load test image
    img = Image.open('test_image.tif')
    
    # Predict
    prediction = model(img)
    ```
    
    **Option 2: Deploy to Cloud**
    - Hugging Face Spaces
    - Streamlit Cloud
    - AWS/GCP with working PyTorch
    """)

with tab3:
    st.markdown("""
    ### Deployment Options
    
    **Cloud Platforms (Recommended):**
    1. **Streamlit Cloud** (free)
       - No PyTorch DLL issues
       - Easy deployment
       - Public URL
    
    2. **Hugging Face Spaces** (free)
       - GPU available
       - Good for ML apps
    
    3. **Google Cloud Run**
       - Scalable
       - Pay per use
    
    **Local (If you fix DLL issue):**
    - Install Anaconda
    - Create isolated environment
    - Install PyTorch from conda
    """)

# Files overview
st.header("📊 Dataset Overview")

if Path("data/raw/S1").exists():
    s1_files = list(Path("data/raw/S1").glob("*.tif"))
    if s1_files:
        st.write(f"**Sample files:**")
        for i, f in enumerate(s1_files[:5], 1):
            st.text(f"{i}. {f.name}")
        if len(s1_files) > 5:
            st.text(f"... and {len(s1_files) - 5} more files")

# Next steps
st.header("🎯 Next Steps")

st.success("""
**Immediate Actions:**

1. ✅ **Training Complete** (if you ran Colab script)
   - Check if model downloaded
   - Copy to models/ folder

2. 🔄 **Test Predictions** (on Colab)
   - Upload test images
   - Run inference
   - Visualize results

3. 🚀 **Deploy** (Optional)
   - Choose cloud platform
   - Upload your code
   - Share public URL

**For Windows PyTorch Issues:**
- Use Google Colab for all PyTorch operations
- Or install Anaconda and create conda environment
- Or deploy to cloud (Streamlit Cloud, etc.)
""")

# Footer
st.markdown("---")
st.markdown("""
### 📖 Documentation
- README.md - Project overview
- SETUP.md - Installation guide
- STATUS.md - Current status
- QUICK_START.md - Getting started

### 🔗 Resources
- [Sen1Floods11 Dataset](https://github.com/cloudtostreet/Sen1Floods11)
- [DeepSARFlood Paper](https://github.com/hydrosenselab/DeepSARFlood)
- [Google Colab](https://colab.research.google.com/)
""")

