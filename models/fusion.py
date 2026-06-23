"""Fusion layer for combining image and text features"""

import torch
import torch.nn as nn


class FusionLayer(nn.Module):
    """Multimodal fusion layer"""
    
    def __init__(self, image_dim=2048, text_dim=768, fusion_type='concat'):
        super().__init__()
        
        self.fusion_type = fusion_type
        self.image_dim = image_dim
        self.text_dim = text_dim
        
        if fusion_type == 'concat':
            self.output_dim = image_dim + text_dim
        elif fusion_type == 'attention':
            self.query = nn.Linear(image_dim, 512)
            self.key = nn.Linear(text_dim, 512)
            self.value = nn.Linear(text_dim, 512)
            self.output_dim = 512
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")
    
    def forward(self, image_features, text_features):
        """Forward pass
        
        Args:
            image_features: (B, image_dim)
            text_features: (B, text_dim)
            
        Returns:
            Fused features
        """
        if self.fusion_type == 'concat':
            return torch.cat([image_features, text_features], dim=1)
        elif self.fusion_type == 'attention':
            q = self.query(image_features)
            k = self.key(text_features)
            v = self.value(text_features)
            weights = torch.softmax(q @ k.t(), dim=-1)
            return weights @ v
