"""
Uncertainty quantification for flood predictions
"""

import numpy as np
from typing import Tuple


def calculate_uncertainty(predictions: np.ndarray) -> np.ndarray:
    """
    Calculate epistemic uncertainty from ensemble predictions
    
    Args:
        predictions: Array of predictions from ensemble (M, H, W)
                    where M is number of models
        
    Returns:
        Uncertainty map (H, W)
    """
    # Calculate variance across models
    uncertainty = np.var(predictions, axis=0)
    
    return uncertainty


def classify_uncertainty(uncertainty: np.ndarray, thresholds: list = [0.2, 0.5, 0.8]) -> np.ndarray:
    """
    Classify uncertainty into categories
    
    Args:
        uncertainty: Uncertainty map
        thresholds: List of thresholds for classification
        
    Returns:
        Classified uncertainty (0: low, 1: medium, 2: high, 3: very high)
    """
    classified = np.zeros_like(uncertainty, dtype=np.int32)
    
    for i, threshold in enumerate(thresholds):
        classified[uncertainty >= threshold] = i + 1
    
    return classified


def generate_uncertainty_map(
    predictions: np.ndarray,
    probabilities: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate uncertainty map from predictions and probabilities
    
    Args:
        predictions: Ensemble predictions (M, H, W)
        probabilities: Ensemble probabilities (M, C, H, W)
        
    Returns:
        tuple: (uncertainty, confidence)
    """
    # Calculate epistemic uncertainty (variance across models)
    epistemic = np.var(predictions, axis=0)
    
    # Calculate aleatoric uncertainty (mean entropy)
    entropy = -np.sum(probabilities * np.log(probabilities + 1e-10), axis=1)
    aleatoric = np.mean(entropy, axis=0)
    
    # Combined uncertainty
    uncertainty = (epistemic + aleatoric) / 2
    
    # Confidence (inverse of uncertainty)
    confidence = 1 / (1 + uncertainty)
    
    return uncertainty, confidence


if __name__ == "__main__":
    # Example usage
    predictions = np.random.randint(0, 2, size=(3, 256, 256))
    probabilities = np.random.rand(3, 2, 256, 256)
    
    uncertainty = calculate_uncertainty(predictions)
    print(f"Uncertainty shape: {uncertainty.shape}")
    print(f"Mean uncertainty: {uncertainty.mean():.4f}")
    
    uncertainty_map, confidence_map = generate_uncertainty_map(predictions, probabilities)
    print(f"Uncertainty map shape: {uncertainty_map.shape}")
    print(f"Confidence map shape: {confidence_map.shape}")

