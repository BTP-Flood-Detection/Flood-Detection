"""
MaxViT (Multi-Axis Vision Transformer) for flood segmentation
Hybrid CNN-ViT architecture adapted for multi-channel input
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import Tuple


class MaxViTFloodSegmentation(nn.Module):
    """
    MaxViT-based segmentation model for flood detection
    Combines CNN and Vision Transformer for multi-scale feature extraction
    """
    
    def __init__(self, input_channels: int = 6, num_classes: int = 2,
                 model_name: str = 'maxvit_base_tf_224', pretrained: bool = True):
        """
        Args:
            input_channels: Number of input channels
            num_classes: Number of output classes
            model_name: MaxViT model variant
            pretrained: Whether to use pretrained weights
        """
        super().__init__()
        
        # Load pretrained MaxViT as backbone
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
            global_pool=''
        )
        
        # Adapt first layer for multi-channel input
        if hasattr(self.backbone, 'stem'):
            # Replace stem convolution
            original_conv = self.backbone.stem[0]
            self.backbone.stem[0] = nn.Conv2d(
                input_channels,
                original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                bias=original_conv.bias is not None
            )
            
            # Initialize new weights
            if pretrained:
                with torch.no_grad():
                    self.backbone.stem[0].weight[:, :3] = original_conv.weight[:, :3]
                    nn.init.kaiming_normal_(self.backbone.stem[0].weight[:, 3:])
        
        # Get feature dimensions
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, 224, 224)
            features = self.backbone(dummy_input)
            if isinstance(features, (list, tuple)):
                feat_dims = features[-1].shape[1]
            else:
                feat_dims = features.shape[1]
        
        # Decoder for segmentation
        self.decoder = nn.Sequential(
            # Upsample progressively
            nn.ConvTranspose2d(feat_dims, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, num_classes, kernel_size=1)
        )
        
        # Use last feature map
        self.last_feat_dim = feat_dims
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor (B, C, H, W)
            
        Returns:
            Output tensor (B, num_classes, H', W')
        """
        # Encoder (MaxViT backbone)
        features = self.backbone(x)
        
        # Handle different feature formats
        if isinstance(features, (list, tuple)):
            features = features[-1]
        
        # Reshape if needed (from (B, N, C) to (B, C, H, W))
        if len(features.shape) == 3:
            B, N, C = features.shape
            H = W = int(N ** 0.5)
            features = features.transpose(1, 2).reshape(B, C, H, W)
        
        # Decoder
        output = self.decoder(features)
        
        # Resize to match input resolution
        if output.shape[2:] != x.shape[2:]:
            output = F.interpolate(output, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        return output


if __name__ == "__main__":
    # Test model
    model = MaxViTFloodSegmentation(input_channels=6, num_classes=2, pretrained=True)
    
    # Test with dummy input
    x = torch.randn(2, 6, 224, 224)
    output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

