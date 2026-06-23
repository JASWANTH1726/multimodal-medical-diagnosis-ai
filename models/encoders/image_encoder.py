"""Image encoder using ResNet or Vision Transformer"""

import torch
import torch.nn as nn
from torchvision import models


class ImageEncoder(nn.Module):
    """Image encoder for medical images"""
    
    def __init__(self, model_name='resnet50', pretrained=True):
        super().__init__()
        
        if model_name == 'resnet50':
            model = models.resnet50(pretrained=pretrained)
            self.encoder = nn.Sequential(*list(model.children())[:-1])
            self.output_dim = 2048
        else:
            raise ValueError(f"Unknown model: {model_name}")
    
    def forward(self, x):
        """Forward pass
        
        Args:
            x: Image tensor (B, 3, H, W)
            
        Returns:
            Feature tensor (B, output_dim)
        """
        features = self.encoder(x)
        features = features.view(features.size(0), -1)
        return features
