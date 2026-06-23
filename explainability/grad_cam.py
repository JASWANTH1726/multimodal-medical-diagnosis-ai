"""Grad-CAM visualization for medical images"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2


class GradCAM:
    """Gradient-weighted Class Activation Mapping"""
    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()
    
    def _register_hooks(self):
        """Register hooks to capture gradients and activations"""
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)
    
    def generate(self, images, class_idx=None):
        """Generate Grad-CAM heatmap
        
        Args:
            images: Input images tensor (B, 3, H, W)
            class_idx: Target class index
            
        Returns:
            Heatmaps (B, H, W)
        """
        self.model.eval()
        
        # Forward pass
        logits = self.model(images, ["dummy_text"] * len(images))
        
        if class_idx is None:
            class_idx = logits.argmax(dim=1)
        
        # Backward pass
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, class_idx.unsqueeze(1), 1)
        logits.backward(gradient=one_hot)
        
        # Compute Grad-CAM
        gradients = self.gradients
        activations = self.activations
        
        b, c, h, w = activations.shape
        cam = torch.zeros(b, h, w).to(activations.device)
        
        for i in range(b):
            grad = gradients[i].mean(dim=(1, 2), keepdim=True)
            weighted_act = (grad * activations[i]).sum(dim=0)
            cam[i] = F.relu(weighted_act)
            cam[i] = (cam[i] - cam[i].min()) / (cam[i].max() - cam[i].min() + 1e-8)
        
        return cam


def visualize_grad_cam(image, heatmap, save_path=None):
    """Visualize Grad-CAM heatmap
    
    Args:
        image: Original image (H, W, 3)
        heatmap: Grad-CAM heatmap (H, W)
        save_path: Optional path to save
        
    Returns:
        Overlaid image
    """
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    overlaid = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)
    
    if save_path:
        cv2.imwrite(save_path, overlaid)
    
    return overlaid
