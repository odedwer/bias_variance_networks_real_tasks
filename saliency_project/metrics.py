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


def face_part_coverage(S, masks, threshold):
    salient = S >= threshold
    total = salient.sum().item()

    if total == 0:
        return {k: 0.0 for k in masks}

    return {
        region: (salient & mask).sum().item() / total
        for region, mask in masks.items()
    }