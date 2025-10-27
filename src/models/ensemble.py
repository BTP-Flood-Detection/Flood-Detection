"""
Ensemble module for combining multiple models with uncertainty quantification
Implements model soups and prediction aggregation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional
import numpy as np

from .resnet_unet import ResNet50UNet
from .swin_transformer import SwinTransformer
from .max_vit import MaxViTFloodSegmentation


class ModelEnsemble(nn.Module):
    """
    Ensemble of multiple models for flood segmentation
    Combines predictions with uncertainty quantification
    """
    
    def __init__(
        self,
        models: List[nn.Module],
        aggregation: str = 'mean',
        uncertainty_threshold: float = 0.5
    ):
        """
        Args:
            models: List of trained models
            aggregation: Aggregation method ('mean', 'median', 'weighted')
            uncertainty_threshold: Threshold for uncertainty calculation
        """
        super().__init__()
        self.models = nn.ModuleList(models)
        self.aggregation = aggregation
        self.uncertainty_threshold = uncertainty_threshold
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through all models
        
        Args:
            x: Input tensor (B, C, H, W)
            
        Returns:
            Ensemble prediction (B, num_classes, H, W)
        """
        predictions = []
        
        for model in self.models:
            with torch.no_grad():
                pred = model(x)
                predictions.append(pred)
        
        # Stack predictions
        pred_stack = torch.stack(predictions, dim=0)  # (M, B, num_classes, H, W)
        
        # Aggregate
        if self.aggregation == 'mean':
            ensemble_pred = pred_stack.mean(dim=0)
        elif self.aggregation == 'median':
            ensemble_pred = pred_stack.median(dim=0)[0]
        elif self.aggregation == 'weighted':
            weights = torch.tensor([1/len(self.models)] * len(self.models)).to(x.device)
            weights = weights.view(-1, 1, 1, 1, 1)
            ensemble_pred = (pred_stack * weights).sum(dim=0)
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation}")
        
        return ensemble_pred
    
    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        return_uncertainty: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Predict with uncertainty quantification
        
        Args:
            x: Input tensor
            return_uncertainty: Whether to return uncertainty map
            
        Returns:
            Dictionary with predictions and uncertainty
        """
        predictions = []
        
        for model in self.models:
            with torch.no_grad():
                pred = model(x)
                # Convert to probabilities
                pred_probs = torch.softmax(pred, dim=1)
                predictions.append(pred_probs)
        
        pred_stack = torch.stack(predictions, dim=0)  # (M, B, num_classes, H, W)
        
        # Calculate mean prediction (epistemic uncertainty)
        mean_pred = pred_stack.mean(dim=0)
        
        # Calculate variance (uncertainty)
        variance = pred_stack.var(dim=0)
        
        # Aggregate uncertainty across channels
        uncertainty = variance.mean(dim=1)  # (B, H, W)
        
        # Final prediction
        final_pred = mean_pred.argmax(dim=1)
        
        result = {
            'prediction': final_pred,
            'probabilities': mean_pred
        }
        
        if return_uncertainty:
            result['uncertainty'] = uncertainty
        
        return result


def create_ensemble(
    num_classes: int = 2,
    input_channels: int = 6,
    pretrained: bool = True
) -> ModelEnsemble:
    """
    Create ensemble of ResNet, Swin, and MaxViT models
    
    Args:
        num_classes: Number of output classes
        input_channels: Number of input channels
        pretrained: Whether to use pretrained weights
        
    Returns:
        ModelEnsemble instance
    """
    models = [
        ResNet50UNet(input_channels=input_channels, num_classes=num_classes, pretrained=pretrained),
        SwinTransformer(input_channels=input_channels, num_classes=num_classes, pretrained=pretrained),
        MaxViTFloodSegmentation(input_channels=input_channels, num_classes=num_classes, pretrained=pretrained)
    ]
    
    return ModelEnsemble(models=models, aggregation='mean')


def load_ensemble(
    model_paths: List[str],
    device: torch.device = torch.device('cpu')
) -> ModelEnsemble:
    """
    Load trained models and create ensemble
    
    Args:
        model_paths: List of paths to model checkpoints
        device: Device to load models on
        
    Returns:
        ModelEnsemble instance
    """
    models = []
    
    for path in model_paths:
        checkpoint = torch.load(path, map_location=device)
        
        # Determine model type from checkpoint
        if 'resnet' in path.lower():
            model = ResNet50UNet(num_classes=checkpoint.get('num_classes', 2))
        elif 'swin' in path.lower():
            model = SwinTransformer(num_classes=checkpoint.get('num_classes', 2))
        elif 'maxvit' in path.lower():
            model = MaxViTFloodSegmentation(num_classes=checkpoint.get('num_classes', 2))
        else:
            raise ValueError(f"Unknown model type in path: {path}")
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        models.append(model)
    
    return ModelEnsemble(models=models)


if __name__ == "__main__":
    # Test ensemble
    ensemble = create_ensemble(num_classes=2, input_channels=6, pretrained=False)
    
    # Test with dummy input
    x = torch.randn(2, 6, 224, 224)
    
    # Forward pass
    output = ensemble(x)
    print(f"Ensemble output shape: {output.shape}")
    
    # Predict with uncertainty
    result = ensemble.predict_with_uncertainty(x)
    print(f"Prediction shape: {result['prediction'].shape}")
    print(f"Uncertainty shape: {result['uncertainty'].shape}")
    print(f"Model parameters: {sum(p.numel() for p in ensemble.parameters()) / 1e6:.2f}M")

