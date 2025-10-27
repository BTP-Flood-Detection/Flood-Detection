"""
Flood Detection Project Dashboard
Completely standalone - no PyTorch required
"""

import streamlit as st
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Flood Detection System",
    page_icon="🌊",
    layout="wide"
)

# Title
st.title("🌊 Flood Detection System - Project Dashboard")

# Project Status
st.header("📊 Project Status")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Code Completion", "100%", "✅")
    st.caption("All modules implemented")

with col2:
    st.metric("Dataset", "446 images", "✅")
    st.caption("Sen1Floods11 downloaded")

with col3:
    st.metric("Models Trained", "1/3", "⚠️")
    st.caption("ResNet on Colab")

# Status Details
st.info("""
**✅ Completed:**
- Complete codebase (25+ Python files)
- Data pipeline and preprocessing
- 3 Model architectures (ResNet, Swin, MaxViT)
- Training pipeline with metrics
- Inference and uncertainty quantification
- Google Earth Engine integration
- Documentation (README, SETUP, etc.)

**⚠️ Note:**
- Windows PyTorch DLL issues prevent local execution
- Solution: Use Google Colab for training and inference
""")

# Check Files
st.header("📁 Project Files")

tab1, tab2, tab3 = st.tabs(["Code", "Data", "Models"])

with tab1:
    st.subheader("Source Code")
    
    code_files = {
        "Data Pipeline": ["src/data/dataset.py", "src/data/preprocessing.py", "src/data/gee_downloader.py"],
        "Models": ["src/models/resnet_unet.py", "src/models/swin_transformer.py", "src/models/max_vit.py", "src/models/ensemble.py"],
        "Training": ["src/training/train.py", "src/training/losses.py", "src/training/metrics.py"],
        "Inference": ["src/inference/predict.py", "src/inference/uncertainty.py"],
        "App": ["app/streamlit_app.py", "app/utils.py"]
    }
    
    for category, files in code_files.items():
        with st.expander(f"**{category}**"):
            for file in files:
                if Path(file).exists():
                    st.success(f"✅ {file}")
                else:
                    st.error(f"❌ {file}")

with tab2:
    st.subheader("Dataset")
    
    data_path = Path("data/raw")
    
    if data_path.exists():
        s1_path = data_path / "S1"
        s2_path = data_path / "S2"
        labels_path = data_path / "labels"
        
        s1_count = len(list(s1_path.glob("*.tif"))) if s1_path.exists() else 0
        s2_count = len(list(s2_path.glob("*.tif"))) if s2_path.exists() else 0
        labels_count = len(list(labels_path.glob("*.tif"))) if labels_path.exists() else 0
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("S1 Images", s1_count)
        with col2:
            st.metric("S2 Images", s2_count)
        with col3:
            st.metric("Labels", labels_count)
        
        if s1_count > 0:
            st.success(f"✅ Dataset ready for training!")
            with st.expander("Sample files"):
                for i, f in enumerate(list(s1_path.glob("*.tif"))[:10], 1):
                    st.text(f"{i}. {f.name}")
    else:
        st.warning("⚠️ Data directory not found")

with tab3:
    st.subheader("Trained Models")
    
    models_path = Path("models")
    
    if models_path.exists():
        model_files = list(models_path.glob("*.pt"))
        
        if model_files:
            st.success(f"✅ Found {len(model_files)} trained model(s)")
            for model in model_files:
                st.write(f"- {model.name}")
        else:
            st.warning("⚠️ No trained models found")
            st.info("Train models using the Colab notebook")
    else:
        st.warning("⚠️ Models directory not found")

# Instructions
st.header("🚀 How to Use This Project")

instructions_tab1, instructions_tab2, instructions_tab3 = st.tabs(["Training", "Inference", "Deployment"])

with instructions_tab1:
    st.markdown("""
    ### Training Models (Google Colab)
    
    **Why Colab?**
    - Free GPU access
    - No Windows DLL issues
    - Pre-configured environment
    
    **Steps:**
    1. Open `COLAB_TRAINING_SINGLE_CELL.py`
    2. Copy all code
    3. Go to https://colab.research.google.com/
    4. Paste into a cell
    5. Change runtime to GPU
    6. Run the cell
    7. Download trained model
    8. Copy to `models/` folder
    
    **Expected Result:**
    - Training time: ~30 minutes with GPU
    - Target IoU: > 0.72
    - Output: `flood_model_best.pt`
    """)
    
    st.code("""
# Quick start in Colab:
# 1. Upload your data to Google Drive
# 2. Copy the training script
# 3. Run and download model
    """, language="python")

with instructions_tab2:
    st.markdown("""
    ### Making Predictions
    
    **Option 1: Google Colab Inference**
    ```python
    import torch
    from PIL import Image
    
    # Load model
    model = torch.load('flood_model_best.pt')
    model.eval()
    
    # Load and predict
    image = Image.open('test_flood.tif')
    prediction = model(preprocess(image))
    ```
    
    **Option 2: Cloud Deployment**
    - Deploy to Streamlit Cloud (no DLL issues)
    - Use Hugging Face Spaces
    - AWS/GCP with proper environment
    """)

with instructions_tab3:
    st.markdown("""
    ### Deployment Options
    
    **1. Streamlit Cloud (Recommended)**
    - Push code to GitHub
    - Connect at https://streamlit.io/cloud
    - Deploy with one click
    - Free tier available
    
    **2. Hugging Face Spaces**
    - Free GPU inference
    - Good for ML models
    - Public URL
    
    **3. Local (Fix DLL Issue)**
    ```bash
    # Install Anaconda
    conda create -n flood python=3.10
    conda activate flood
    conda install pytorch torchvision -c pytorch
    pip install -r requirements.txt
    streamlit run app/streamlit_app.py
    ```
    """)

# Resources
st.header("📚 Resources")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Documentation")
    docs = ["README.md", "SETUP.md", "STATUS.md", "QUICK_START.md", "PROJECT_SUMMARY.md"]
    for doc in docs:
        if Path(doc).exists():
            st.success(f"✅ {doc}")

with col2:
    st.subheader("Notebooks")
    notebooks = ["download_sen1floods11_colab.ipynb", "train_flood_detection_colab.ipynb", "COLAB_TRAINING_SINGLE_CELL.py"]
    for nb in notebooks:
        if Path(nb).exists():
            st.success(f"✅ {nb}")

# Next Steps
st.header("🎯 Next Steps")

st.success("""
**Immediate Actions:**

1. **Test Your Trained Model** (if training is complete)
   - Download `flood_model_best.pt` from Colab
   - Copy to `models/resnet_best.pt`
   - Use Colab for inference

2. **Train Additional Models**
   - Swin Transformer
   - MaxViT
   - Create ensemble for better accuracy

3. **Deploy to Cloud**
   - Push to GitHub
   - Deploy on Streamlit Cloud
   - Share public URL

**For Better Accuracy:**
- Use all 3 models in ensemble
- Target: IoU > 0.72 (state-of-the-art)
- Fine-tune hyperparameters
""")

# Footer
st.markdown("---")
st.caption("Flood Detection System | Based on DeepSARFlood Architecture")

# Sidebar info
with st.sidebar:
    st.header("Quick Links")
    st.markdown("""
    - [Sen1Floods11 Dataset](https://github.com/cloudtostreet/Sen1Floods11)
    - [DeepSARFlood Paper](https://github.com/hydrosenselab/DeepSARFlood)
    - [Google Colab](https://colab.research.google.com/)
    - [Streamlit Cloud](https://streamlit.io/cloud)
    """)
    
    st.header("Contact")
    st.info("""
    For issues or questions:
    - Check documentation
    - Review STATUS.md
    - Use Google Colab for PyTorch
    """)

