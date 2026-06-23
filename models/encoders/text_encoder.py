"""Text encoder using BioBERT or SciBERT"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel


class TextEncoder(nn.Module):
    """Text encoder for clinical text"""
    
    def __init__(self, model_name='dmis-lab/biobert-v1.1'):
        super().__init__()
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.output_dim = 768
    
    def forward(self, text_list):
        """Forward pass
        
        Args:
            text_list: List of text strings
            
        Returns:
            Feature tensor (B, output_dim)
        """
        encoded = self.tokenizer(
            text_list,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        
        outputs = self.model(**encoded)
        # Use [CLS] token representation
        features = outputs.last_hidden_state[:, 0, :]
        return features
