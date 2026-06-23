"""End-to-end diagnosis model"""

import torch
import torch.nn as nn
from models.encoders.image_encoder import ImageEncoder
from models.encoders.text_encoder import TextEncoder
from models.fusion import FusionLayer


class DiagnosisModel(nn.Module):
    """Complete multimodal diagnosis model"""
    
    def __init__(self, num_classes=5, fusion_type='concat', image_model='resnet50'):
        super().__init__()
        
        self.image_encoder = ImageEncoder(image_model)
        self.text_encoder = TextEncoder()
        self.fusion = FusionLayer(
            image_dim=self.image_encoder.output_dim,
            text_dim=self.text_encoder.output_dim,
            fusion_type=fusion_type
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.fusion.output_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        self.num_classes = num_classes
    
    def forward(self, images, texts):
        """Forward pass
        
        Args:
            images: Image tensor (B, 3, H, W)
            texts: List of text strings
            
        Returns:
            Logits (B, num_classes)
        """
        image_features = self.image_encoder(images)
        text_features = self.text_encoder(texts)
        
        fused_features = self.fusion(image_features, text_features)
        logits = self.classifier(fused_features)
        
        return logits
