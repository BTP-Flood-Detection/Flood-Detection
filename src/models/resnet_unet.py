"""
ResNet-50 UNet architecture for flood segmentation
Adapted from DeepSARFlood for multi-channel input
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights


class ResNet50UNet(nn.Module):
    """
    UNet with ResNet-50 encoder for flood segmentation
    Adapted for multi-channel input (S1 + S2 bands)
    """
    
    def __init__(self, input_channels: int = 6, num_classes: int = 2, pretrained: bool = True):
        """
        Args:
            input_channels: Number of input channels (S1: 2, S2: 4)
            num_classes: Number of output classes (2 for binary segmentation)
            pretrained: Whether to use pretrained ResNet-50 weights
        """
        super().__init__()
        
        # Load pretrained ResNet-50
        resnet = resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)
        
        # Replace first conv layer for multi-channel input
        self.encoder1 = nn.Conv2d(
            input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        
        # Encoder (ResNet-50 layers)
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        self.encoder2 = resnet.layer1  # 256 channels
        self.encoder3 = resnet.layer2  # 512 channels
        self.encoder4 = resnet.layer3  # 1024 channels
        self.encoder5 = resnet.layer4  # 2048 channels
        
        # Decoder
        self.decoder5 = self._make_decoder(2048, 1024, 512)
        self.decoder4 = self._make_decoder(512, 512, 256)
        self.decoder3 = self._make_decoder(256, 256, 128)
        self.decoder2 = self._make_decoder(128, 128, 64)
        self.decoder1 = self._make_decoder(64, 64, 64)
        
        # Final classifier
        self.final = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, kernel_size=1)
        )
    
    def _make_decoder(self, in_channels: int, mid_channels: int, out_channels: int) -> nn.Module:
        """Create decoder block"""
        return nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor (B, C, H, W) where C = input_channels
            
        Returns:
            Output tensor (B, num_classes, H, W)
        """
        # Encoder
        e1 = self.encoder1(x)
        e1 = self.bn1(e1)
        e1 = self.relu(e1)
        e1 = self.maxpool(e1)
        
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)
        
        # Decoder with skip connections
        d5 = F.interpolate(e5, size=e4.shape[2:], mode='bilinear', align_corners=False)
        d5 = torch.cat([d5, e4], dim=1)
        d5 = self.decoder5(d5)
        
        d4 = F.interpolate(d5, size=e3.shape[2:], mode='bilinear', align_corners=False)
        d4 = torch.cat([d4, e3], dim=1)
        d4 = self.decoder4(d4)
        
        d3 = F.interpolate(d4, size=e2.shape[2:], mode='bilinear', align_corners=False)
        d3 = torch.cat([d3, e2], dim=1)
        d3 = self.decoder3(d3)
        
        d2 = F.interpolate(d3, size=e1.shape[2:], mode='bilinear', align_corners=False)
        d2 = torch.cat([d2, e1], dim=1)
        d2 = self.decoder2(d2)
        
        d1 = F.interpolate(d2, scale_factor=2, mode='bilinear', align_corners=False)
        d1 = self.decoder1(d1)
        
        # Final classification
        out = self.final(d1)
        
        return out


if __name__ == "__main__":
    # Test model
    model = ResNet50UNet(input_channels=6, num_classes=2, pretrained=True)
    
    # Test with dummy input
    x = torch.randn(2, 6, 512, 512)
    output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

