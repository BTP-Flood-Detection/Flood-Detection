"""
Swin Transformer for flood segmentation
Based on DeepSARFlood architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import Tuple


class SwinTransformer(nn.Module):
    """
    Swin Transformer for semantic segmentation with multi-channel input
    """
    
    def __init__(self, input_channels: int = 6, num_classes: int = 2, 
                 model_name: str = 'swin_base_patch4_window7_224', 
                 pretrained: bool = True):
        """
        Args:
            input_channels: Number of input channels
            num_classes: Number of output classes
            model_name: Swin model variant
            pretrained: Whether to use pretrained weights
        """
        super().__init__()
        
        # Load pretrained Swin Transformer as backbone
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
            global_pool=''
        )
        
        # Get feature dimensions from backbone
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 224, 224)
            features = self.backbone(dummy_input)
            feat_dims = features.shape[1]  # Channel dimension
        
        # Adapt first layer for multi-channel input
        if hasattr(self.backbone, 'patch_embed'):
            # Patch embedding layer
            patch_embed = self.backbone.patch_embed
            # Replace first conv layer
            original_conv = patch_embed.proj
            self.backbone.patch_embed.proj = nn.Conv2d(
                input_channels,
                original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                bias=original_conv.bias is not None
            )
            
            # Initialize new weights
            if pretrained:
                # Copy pretrained weights for RGB channels
                with torch.no_grad():
                    self.backbone.patch_embed.proj.weight[:, :3] = original_conv.weight[:, :3]
                    # Initialize remaining channels
                    nn.init.kaiming_normal_(self.backbone.patch_embed.proj.weight[:, 3:])
        
        # Decoder for segmentation
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(feat_dims, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, num_classes, kernel_size=1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor (B, C, H, W)
            
        Returns:
            Output tensor (B, num_classes, H', W')
        """
        # Encoder (Swin Transformer backbone)
        features = self.backbone(x)
        
        # Reshape features from (B, N, C) to (B, C, H, W)
        # Assuming features are from patch embeddings
        B, N, C = features.shape
        
        # Calculate spatial dimensions (assuming input was 224x224)
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
    model = SwinTransformer(input_channels=6, num_classes=2, pretrained=True)
    
    # Test with dummy input
    x = torch.randn(2, 6, 224, 224)
    output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

