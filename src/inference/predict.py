"""
Inference pipeline for flood detection
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import rasterio
from torchvision import transforms

from ..models.ensemble import ModelEnsemble, load_ensemble
from ..data.preprocessing import stack_s1_s2, normalize_bands, quality_control


class FloodPredictor:
    """Flood prediction pipeline"""
    
    def __init__(
        self,
        model_paths: list,
        device: torch.device = torch.device('cpu'),
        confidence_threshold: float = 0.5
    ):
        """
        Args:
            model_paths: List of paths to trained model checkpoints
            device: Device for inference
            confidence_threshold: Threshold for binary classification
        """
        self.device = device
        self.confidence_threshold = confidence_threshold
        
        # Load ensemble
        self.ensemble = load_ensemble(model_paths, device=device)
        self.ensemble.eval()
        
        # Setup transform
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
    
    def predict_from_tif(
        self,
        s1_path: str,
        s2_path: str,
        return_probabilities: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Predict flood from GeoTIFF files
        
        Args:
            s1_path: Path to Sentinel-1 GeoTIFF
            s2_path: Path to Sentinel-2 GeoTIFF
            return_probabilities: Whether to return probability map
            
        Returns:
            tuple: (prediction, probabilities)
        """
        # Load and preprocess images
        image = self._preprocess_images(s1_path, s2_path)
        
        # Predict
        pred, probs = self._predict(image, return_probabilities)
        
        return pred, probs
    
    def predict_from_array(
        self,
        image: np.ndarray,
        return_probabilities: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Predict flood from numpy array
        
        Args:
            image: Input image array (C, H, W)
            return_probabilities: Whether to return probability map
            
        Returns:
            tuple: (prediction, probabilities)
        """
        # Preprocess
        image = quality_control(image)
        image = normalize_bands(image)
        
        # Predict
        pred, probs = self._predict(image, return_probabilities)
        
        return pred, probs
    
    def _preprocess_images(self, s1_path: str, s2_path: str) -> np.ndarray:
        """Preprocess S1 and S2 images"""
        # Stack bands
        image = stack_s1_s2(s1_path, s2_path, s1_bands=2, s2_bands=4)
        
        # Quality control
        image = quality_control(image)
        
        # Normalize
        image = normalize_bands(image)
        
        return image
    
    def _predict(
        self,
        image: np.ndarray,
        return_probabilities: bool
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Run inference"""
        # Convert to tensor
        image_tensor = torch.from_numpy(image).float().unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            result = self.ensemble.predict_with_uncertainty(image_tensor, return_uncertainty=True)
        
        # Extract prediction and probabilities
        pred = result['prediction'].cpu().numpy()[0]
        probs = result['probabilities'].cpu().numpy()[0] if return_probabilities else None
        
        return pred, probs
    
    def predict_with_uncertainty(
        self,
        s1_path: str,
        s2_path: str
    ) -> dict:
        """
        Predict with uncertainty quantification
        
        Returns:
            Dictionary with 'prediction', 'probabilities', 'uncertainty'
        """
        # Load and preprocess
        image = self._preprocess_images(s1_path, s2_path)
        
        # Convert to tensor
        image_tensor = torch.from_numpy(image).float().unsqueeze(0).to(self.device)
        
        # Predict with uncertainty
        with torch.no_grad():
            result = self.ensemble.predict_with_uncertainty(image_tensor, return_uncertainty=True)
        
        return {
            'prediction': result['prediction'].cpu().numpy()[0],
            'probabilities': result['probabilities'].cpu().numpy()[0],
            'uncertainty': result['uncertainty'].cpu().numpy()[0]
        }


if __name__ == "__main__":
    # Example usage
    model_paths = [
        "models/resnet_best.pt",
        "models/swin_best.pt",
        "models/maxvit_best.pt"
    ]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    predictor = FloodPredictor(model_paths, device=device)
    
    # Predict
    s1_path = "data/raw/S1/test.tif"
    s2_path = "data/raw/S2/test.tif"
    
    pred, probs = predictor.predict_from_tif(s1_path, s2_path)
    print(f"Prediction shape: {pred.shape}")
    print(f"Probabilities shape: {probs.shape}")

