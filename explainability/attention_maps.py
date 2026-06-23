"""Attention map visualization"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import get_cmap


def extract_attention_maps(model, images):
    """Extract attention maps from model
    
    Args:
        model: Diagnosis model
        images: Input images tensor (B, 3, H, W)
        
    Returns:
        Attention maps tensor
    """
    model.eval()
    with torch.no_grad():
        # Extract features from image encoder
        image_features = model.image_encoder(images)
    return image_features


def visualize_attention(attention_map, original_image, save_path=None):
    """Visualize attention map overlaid on image
    
    Args:
        attention_map: Attention weights
        original_image: Original image tensor (3, H, W)
        save_path: Optional path to save visualization
        
    Returns:
        Figure with visualization
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Original image
    if isinstance(original_image, torch.Tensor):
        original_image = original_image.cpu().numpy().transpose(1, 2, 0)
    axes[0].imshow(original_image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Attention map
    if isinstance(attention_map, torch.Tensor):
        attention_map = attention_map.cpu().numpy()
    axes[1].imshow(original_image)
    axes[1].imshow(attention_map, cmap='hot', alpha=0.5)
    axes[1].set_title('Attention Map')
    axes[1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig
