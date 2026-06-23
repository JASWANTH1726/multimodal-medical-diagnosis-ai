"""Tests for model components"""

import pytest
import torch
from models.encoders.image_encoder import ImageEncoder
from models.encoders.text_encoder import TextEncoder
from models.fusion import FusionLayer
from models.diagnosis_model import DiagnosisModel


class TestImageEncoder:
    """Test suite for image encoder"""
    
    def test_image_encoder_initialization(self):
        """Test image encoder initialization"""
        encoder = ImageEncoder('resnet50')
        assert encoder.output_dim == 2048
    
    def test_image_encoder_output_shape(self):
        """Test that image encoder produces correct output shape"""
        encoder = ImageEncoder('resnet50')
        images = torch.randn(4, 3, 224, 224)
        output = encoder(images)
        assert output.shape == (4, 2048)
    
    def test_image_encoder_with_batch(self):
        """Test image encoder with different batch sizes"""
        encoder = ImageEncoder('resnet50')
        for batch_size in [1, 8, 16]:
            images = torch.randn(batch_size, 3, 224, 224)
            output = encoder(images)
            assert output.shape == (batch_size, 2048)


class TestTextEncoder:
    """Test suite for text encoder"""
    
    def test_text_encoder_initialization(self):
        """Test text encoder initialization"""
        encoder = TextEncoder()
        assert encoder.output_dim == 768
    
    def test_text_encoder_output_shape(self):
        """Test that text encoder produces correct output shape"""
        encoder = TextEncoder()
        texts = ["Patient with fever", "Chest pain symptoms"]
        output = encoder(texts)
        assert output.shape == (2, 768)
    
    def test_text_encoder_single_text(self):
        """Test text encoder with single text"""
        encoder = TextEncoder()
        text = ["Patient symptoms"]
        output = encoder(text)
        assert output.shape == (1, 768)


class TestFusion:
    """Test suite for fusion layer"""
    
    def test_fusion_concatenation(self):
        """Test concatenation fusion strategy"""
        fusion = FusionLayer(2048, 768, 'concat')
        image_features = torch.randn(4, 2048)
        text_features = torch.randn(4, 768)
        output = fusion(image_features, text_features)
        assert output.shape == (4, 2816)
    
    def test_fusion_attention(self):
        """Test attention-based fusion strategy"""
        fusion = FusionLayer(2048, 768, 'attention')
        image_features = torch.randn(4, 2048)
        text_features = torch.randn(4, 768)
        output = fusion(image_features, text_features)
        assert output.shape == (4, 512)


class TestDiagnosisModel:
    """Test suite for complete diagnosis model"""
    
    def test_model_initialization(self):
        """Test model initialization"""
        model = DiagnosisModel(num_classes=5)
        assert model.num_classes == 5
    
    def test_model_forward_pass(self):
        """Test model forward pass"""
        model = DiagnosisModel(num_classes=5)
        images = torch.randn(4, 3, 224, 224)
        texts = ["Patient symptoms"] * 4
        output = model(images, texts)
        assert output.shape == (4, 5)
    
    def test_model_different_batch_sizes(self):
        """Test model with different batch sizes"""
        model = DiagnosisModel(num_classes=5)
        for batch_size in [1, 8, 16]:
            images = torch.randn(batch_size, 3, 224, 224)
            texts = ["Patient symptoms"] * batch_size
            output = model(images, texts)
            assert output.shape == (batch_size, 5)
