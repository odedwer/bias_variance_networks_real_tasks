# saliency/compute_saliency.py
"""
Saliency map computation using gradient-based attribution.

Computes gradient-based saliency maps for visualizing which input features
most influence model predictions.
"""

import torch


def compute_saliency(model, image, target_class):
    """
    Compute gradient-based saliency map for a model prediction.
    
    Uses the gradient of the target class output with respect to input
    to identify which pixels most influence the prediction.
    
    Args:
        model: Neural network model (should be in eval mode)
        image (torch.Tensor): Input image tensor of shape (C, H, W)
        target_class (int): Class index to compute saliency for
        
    Returns:
        torch.Tensor: Normalized saliency map of shape (H, W) with values in [0, 1]
    """
    image = image.unsqueeze(0).requires_grad_(True)  # (1,C,H,W)
    output = model(image)

    score = output[0, target_class]
    score.backward()

    saliency = image.grad.abs().squeeze(0)   # (C,H,W)
    saliency = saliency.max(dim=0)[0]         # (H,W)

    saliency = saliency / (saliency.max() + 1e-12) #normalize to [0,1]
    return saliency.detach()