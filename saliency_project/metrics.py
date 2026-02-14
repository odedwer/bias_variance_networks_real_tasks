# metrics/saliency_metrics.py

import torch
import numpy as np
from scipy.spatial import cKDTree


def saliency_entropy(S):
    s = S.flatten()
    s = s / (s.sum() + 1e-12)
    return float(-(s * (s + 1e-12).log()).sum())


def max_short_distance(S, threshold):
    coords = (S >= threshold).nonzero(as_tuple=False)
    if coords.shape[0] < 2:
        return 0.0

    points = coords.cpu().numpy()
    tree = cKDTree(points)
    dists, _ = tree.query(points, k=2)
    return float(dists[:, 1].max())

def mean_short_distance(S, threshold):
    coords = (S >= threshold).nonzero(as_tuple=False)
    if coords.shape[0] < 2:
        return 0.0

    points = coords.cpu().numpy()
    tree = cKDTree(points)
    dists, _ = tree.query(points, k=2)
    return float(dists[:, 1].mean())

def top_k_concentration(S, k=0.1):
    """
    What fraction of total saliency is in top k% of pixels?
    Higher = more local, Lower = more global
    """
    s = S.flatten()
    n_top = int(len(s) * k)
    top_vals = s.topk(n_top)[0]
    return float(top_vals.sum() / (s.sum() + 1e-12))

def face_part_coverage(S, masks, threshold):
    """
    Compute what percentage of each face region is salient.
    
    Args:
        S: Saliency map (H, W) tensor
        masks: dict of boolean masks for each region
        threshold: saliency threshold
    
    Returns:
        dict: {region_name: percentage of that region that is salient}
    """
    salient = S >= threshold
    
    coverage = {}
    for region, mask in masks.items():
        mask_size = mask.sum().item()
        if mask_size == 0:
            coverage[region] = 0.0
        else:
            salient_in_region = (salient & mask).sum().item()
            coverage[f"coverage_{region}"] = salient_in_region / mask_size
    
    return coverage


def saliency_attribution(S, masks):
    """
    Compute what percentage of total saliency comes from each region.
    
    Args:
        S: Saliency map (H, W) tensor
        masks: dict of boolean masks for each region
    
    Returns:
        dict: {region_name: fraction of total saliency from this region}
    """
    total_saliency = S.sum().item()
    
    if total_saliency == 0:
        return {k: 0.0 for k in masks}
    
    attribution = {}
    for region, mask in masks.items():
        saliency_in_region = (S * mask).sum().item()
        attribution[f"attribution_{region}"] = saliency_in_region / total_saliency
    
    return attribution