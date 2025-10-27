"""
Metrics for evaluating flood segmentation models
IoU, F1, Precision, Recall, etc.
"""

import torch
import numpy as np
from typing import Dict


def calculate_iou(pred: torch.Tensor, target: torch.Tensor, num_classes: int = 2) -> torch.Tensor:
    """
    Calculate Intersection over Union (IoU) for each class
    
    Args:
        pred: Predicted logits (B, C, H, W) or predicted labels (B, H, W)
        target: Ground truth labels (B, H, W)
        num_classes: Number of classes
        
    Returns:
        IoU per class (C,)
    """
    # Convert to class predictions if needed
    if len(pred.shape) == 4:
        pred = torch.argmax(pred, dim=1)
    
    pred = pred.flatten()
    target = target.flatten().long()
    
    ious = []
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)
        
        intersection = (pred_cls & target_cls).float().sum()
        union = (pred_cls | target_cls).float().sum()
        
        if union > 0:
            iou = intersection / union
        else:
            iou = torch.tensor(1.0 if intersection == 0 else 0.0, device=pred.device)
        
        ious.append(iou)
    
    return torch.tensor(ious)


def calculate_f1_score(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Calculate F1 score
    
    Args:
        pred: Predicted logits (B, C, H, W) or labels (B, H, W)
        target: Ground truth labels (B, H, W)
        
    Returns:
        F1 score
    """
    # Convert to binary predictions
    if len(pred.shape) == 4:
        pred = torch.argmax(pred, dim=1)
    
    pred = pred.flatten().long()
    target = target.flatten().long()
    
    # Binary classification: flood vs background
    tp = ((pred == 1) & (target == 1)).sum().item()
    fp = ((pred == 1) & (target == 0)).sum().item()
    fn = ((pred == 0) & (target == 1)).sum().item()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return f1


def calculate_precision_recall(pred: torch.Tensor, target: torch.Tensor) -> tuple:
    """
    Calculate Precision and Recall
    
    Args:
        pred: Predicted labels (B, H, W)
        target: Ground truth labels (B, H, W)
        
    Returns:
        (precision, recall)
    """
    # Convert to binary predictions
    if len(pred.shape) == 4:
        pred = torch.argmax(pred, dim=1)
    
    pred = pred.flatten().long()
    target = target.flatten().long()
    
    tp = ((pred == 1) & (target == 1)).sum().item()
    fp = ((pred == 1) & (target == 0)).sum().item()
    fn = ((pred == 0) & (target == 1)).sum().item()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    return precision, recall


def calculate_pixel_accuracy(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Calculate pixel accuracy
    
    Args:
        pred: Predicted labels (B, H, W)
        target: Ground truth labels (B, H, W)
        
    Returns:
        Pixel accuracy
    """
    # Convert to class predictions if needed
    if len(pred.shape) == 4:
        pred = torch.argmax(pred, dim=1)
    
    correct = (pred == target).float()
    accuracy = correct.mean().item()
    
    return accuracy


def calculate_all_metrics(pred: torch.Tensor, target: torch.Tensor, num_classes: int = 2) -> Dict[str, float]:
    """
    Calculate all metrics
    
    Args:
        pred: Predicted logits (B, C, H, W) or labels (B, H, W)
        target: Ground truth labels (B, H, W)
        num_classes: Number of classes
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # IoU per class
    iou = calculate_iou(pred, target, num_classes)
    for cls_idx in range(num_classes):
        metrics[f'iou_class_{cls_idx}'] = iou[cls_idx].item()
    
    # Mean IoU
    metrics['mean_iou'] = iou.mean().item()
    
    # F1, Precision, Recall
    metrics['f1_score'] = calculate_f1_score(pred, target)
    precision, recall = calculate_precision_recall(pred, target)
    metrics['precision'] = precision
    metrics['recall'] = recall
    
    # Pixel accuracy
    metrics['pixel_accuracy'] = calculate_pixel_accuracy(pred, target)
    
    return metrics


if __name__ == "__main__":
    # Test metrics
    pred = torch.randn(2, 2, 512, 512)
    target = torch.randint(0, 2, (2, 512, 512))
    
    metrics = calculate_all_metrics(pred, target)
    print("Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

