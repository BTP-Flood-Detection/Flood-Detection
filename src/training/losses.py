"""
Loss functions for flood segmentation
Dice loss, Focal loss, and combined losses
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Dice Loss for segmentation"""
    
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predicted logits (B, C, H, W)
            target: Ground truth labels (B, H, W)
        """
        # Convert to one-hot
        num_classes = pred.shape[1]
        target_one_hot = F.one_hot(target.long(), num_classes).permute(0, 3, 1, 2).float()
        
        # Apply softmax to predictions
        pred_soft = F.softmax(pred, dim=1)
        
        # Calculate Dice coefficient for each class
        intersection = (pred_soft * target_one_hot).sum(dim=(2, 3))
        union = pred_soft.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
        
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        
        # Return average dice loss across all classes
        return 1 - dice.mean()


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predicted logits (B, C, H, W)
            target: Ground truth labels (B, H, W)
        """
        ce_loss = F.cross_entropy(pred, target.long(), reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        return focal_loss.mean()


class CombinedLoss(nn.Module):
    """Combined Dice and BCE Loss"""
    
    def __init__(self, dice_weight: float = 0.5, bce_weight: float = 0.5):
        super().__init__()
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.CrossEntropyLoss()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Combined loss"""
        dice = self.dice_loss(pred, target)
        bce = self.bce_loss(pred, target.long())
        
        return self.dice_weight * dice + self.bce_weight * bce


class DiceBCELossWithFocal(nn.Module):
    """Combined Dice, BCE, and Focal Loss for optimal training"""
    
    def __init__(self, dice_weight: float = 0.4, bce_weight: float = 0.4, focal_weight: float = 0.2):
        super().__init__()
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.CrossEntropyLoss()
        self.focal_loss = FocalLoss()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.focal_weight = focal_weight
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Combined loss"""
        dice = self.dice_loss(pred, target)
        bce = self.bce_loss(pred, target.long())
        focal = self.focal_loss(pred, target)
        
        return (self.dice_weight * dice + 
                self.bce_weight * bce + 
                self.focal_weight * focal)


def get_loss_function(loss_type: str = 'dice_bce') -> nn.Module:
    """
    Get loss function by type
    
    Args:
        loss_type: Type of loss ('dice', 'focal', 'dice_bce', 'combined')
        
    Returns:
        Loss function module
    """
    if loss_type == 'dice':
        return DiceLoss()
    elif loss_type == 'focal':
        return FocalLoss()
    elif loss_type == 'dice_bce':
        return CombinedLoss(dice_weight=0.5, bce_weight=0.5)
    elif loss_type == 'combined':
        return DiceBCELossWithFocal()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


if __name__ == "__main__":
    # Test loss functions
    pred = torch.randn(2, 2, 512, 512)
    target = torch.randint(0, 2, (2, 512, 512))
    
    # Test Dice loss
    dice_loss = DiceLoss()
    loss = dice_loss(pred, target)
    print(f"Dice Loss: {loss.item():.4f}")
    
    # Test Combined loss
    combined_loss = CombinedLoss()
    loss = combined_loss(pred, target)
    print(f"Combined Loss: {loss.item():.4f}")
    
    # Test Focal loss
    focal_loss = FocalLoss()
    loss = focal_loss(pred, target)
    print(f"Focal Loss: {loss.item():.4f}")

