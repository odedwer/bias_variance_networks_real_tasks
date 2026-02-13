# saliency/compute_saliency.py

import torch


def compute_saliency(model, image, target_class):
    """
    Standard gradient-based saliency map.
    Returns a single-channel normalized saliency map.
    """

    image = image.unsqueeze(0).requires_grad_(True)  # (1,C,H,W)
    output = model(image)

    score = output[0, target_class]
    score.backward()

    saliency = image.grad.abs().squeeze(0)   # (C,H,W)
    saliency = saliency.max(dim=0)[0]         # (H,W)

    saliency = saliency / (saliency.max() + 1e-12)
    return saliency.detach()