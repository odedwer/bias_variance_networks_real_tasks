# metrics/saliency_metrics.py
"""
Saliency map analysis and feature attribution metrics.

Provides functions for:
- Computing saliency entropy (concentration of attention)
- Cluster analysis (MDL, DBSCAN, connected components)
- Face part coverage and attribution analysis
"""

import torch
import numpy as np
from scipy.spatial import cKDTree


def saliency_entropy(S):
    s = S.flatten()
    s = s / (s.sum() + 1e-12) #normalize pixels to sum to 1
    H = -(s * (s + 1e-12).log()).sum()

    # Normalize by log(N) where N is number of pixels, so that metric is in [0,1]
    N = s.numel()
    H_norm = H / np.log(N)

    return H_norm.item()


def maxmean_short_distance(S, threshold):
    coords = (S >= threshold).nonzero(as_tuple=False)
    if coords.shape[0] < 2:
        return 0.0

    points = coords.cpu().numpy()
    tree = cKDTree(points)
    dists, _ = tree.query(points, k=2)
    return float(dists[:, 1].max()), float(dists[:, 1].mean())

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

# metrics.py

from sklearn.cluster import AgglomerativeClustering, DBSCAN
from scipy.spatial.distance import pdist, squareform
from scipy.ndimage import label as connected_components
import numpy as np

def mdl_cluster_analysis(S, threshold, max_clusters=10):
    """
    Use MDL principle to find optimal clustering of saliency regions.
    
    Returns metrics about cluster structure with threshold in key names.
    """
    # Get high-saliency coordinates
    coords = (S >= threshold).nonzero(as_tuple=False).cpu().numpy()
    
    thresh_str = f"thresh_{threshold:.1f}"
    
    if len(coords) < 2:
        return {
            f'mdl_num_clusters_{thresh_str}': 0,
            f'mdl_avg_inter_cluster_distance_{thresh_str}': 0.0,
            f'mdl_max_inter_cluster_distance_{thresh_str}': 0.0,
            f'mdl_cluster_size_std_{thresh_str}': 0.0,
            f'mdl_score_{thresh_str}': 0.0
        }
    
    # Find optimal number of clusters using MDL
    best_n_clusters = 1
    best_mdl = float('inf')
    
    for n in range(2, min(max_clusters + 1, len(coords))):
        clustering = AgglomerativeClustering(n_clusters=n, linkage='ward')
        labels = clustering.fit_predict(coords)
        
        # Calculate MDL: description length = data cost + model cost
        # Data cost: within-cluster variance
        data_cost = 0
        for cluster_id in range(n):
            cluster_points = coords[labels == cluster_id]
            if len(cluster_points) > 0:
                centroid = cluster_points.mean(axis=0)
                data_cost += np.sum((cluster_points - centroid) ** 2)
        
        # Model cost: number of parameters (penalize complex models)
        model_cost = n * 3  # 3 parameters per cluster
        
        mdl = data_cost + model_cost * np.log(len(coords))
        
        if mdl < best_mdl:
            best_mdl = mdl
            best_n_clusters = n
    
    # Re-cluster with optimal number
    clustering = AgglomerativeClustering(n_clusters=best_n_clusters, linkage='ward')
    labels = clustering.fit_predict(coords)
    
    # Calculate cluster centers
    cluster_centers = []
    cluster_sizes = []
    for cluster_id in range(best_n_clusters):
        cluster_points = coords[labels == cluster_id]
        if len(cluster_points) > 0:
            cluster_centers.append(cluster_points.mean(axis=0))
            cluster_sizes.append(len(cluster_points))
    
    cluster_centers = np.array(cluster_centers)
    
    # Calculate inter-cluster distances
    if len(cluster_centers) > 1:
        distances = pdist(cluster_centers, metric='euclidean')
        avg_distance = distances.mean()
        max_distance = distances.max()
    else:
        avg_distance = 0.0
        max_distance = 0.0
    
    # Cluster size variation
    cluster_size_std = np.std(cluster_sizes) if len(cluster_sizes) > 1 else 0.0
    
    return {
        f'mdl_num_clusters_{thresh_str}': best_n_clusters,
        f'mdl_avg_inter_cluster_distance_{thresh_str}': float(avg_distance),
        f'mdl_max_inter_cluster_distance_{thresh_str}': float(max_distance),
        f'mdl_cluster_size_std_{thresh_str}': float(cluster_size_std),
        f'mdl_score_{thresh_str}': float(best_mdl)
    }


def dbscan_cluster_analysis(S, threshold, eps=5.0, min_samples=10):
    """
    Use DBSCAN to find natural clusters in saliency.
    Automatically determines number of clusters.
    """
    # Get high-saliency coordinates
    coords = (S >= threshold).nonzero(as_tuple=False).cpu().numpy()
    
    thresh_str = f"thresh_{threshold:.1f}"
    
    if len(coords) < min_samples:
        return {
            f'dbscan_num_clusters_{thresh_str}': 0,
            f'dbscan_avg_inter_cluster_distance_{thresh_str}': 0.0,
            f'dbscan_max_inter_cluster_distance_{thresh_str}': 0.0,
            f'dbscan_cluster_compactness_{thresh_str}': 0.0,
            f'dbscan_noise_ratio_{thresh_str}': 1.0
        }
    
    # DBSCAN clustering
    clustering = DBSCAN(eps=eps, min_samples=min_samples)
    labels = clustering.fit_predict(coords)
    
    # Number of clusters (excluding noise, labeled as -1)
    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise_points = np.sum(labels == -1)
    noise_ratio = noise_points / len(labels) if len(labels) > 0 else 0
    
    if num_clusters == 0:
        return {
            f'dbscan_num_clusters_{thresh_str}': 0,
            f'dbscan_avg_inter_cluster_distance_{thresh_str}': 0.0,
            f'dbscan_max_inter_cluster_distance_{thresh_str}': 0.0,
            f'dbscan_cluster_compactness_{thresh_str}': 0.0,
            f'dbscan_noise_ratio_{thresh_str}': noise_ratio
        }
    
    # Calculate cluster centers and compactness
    cluster_centers = []
    cluster_compactness_values = []
    
    for cluster_id in range(num_clusters):
        cluster_points = coords[labels == cluster_id]
        if len(cluster_points) > 0:
            centroid = cluster_points.mean(axis=0)
            cluster_centers.append(centroid)
            
            # Compactness: average distance from centroid
            distances = np.linalg.norm(cluster_points - centroid, axis=1)
            cluster_compactness_values.append(distances.mean())
    
    cluster_centers = np.array(cluster_centers)
    
    # Inter-cluster distances
    if len(cluster_centers) > 1:
        distances = pdist(cluster_centers, metric='euclidean')
        avg_distance = distances.mean()
        max_distance = distances.max()
    else:
        avg_distance = 0.0
        max_distance = 0.0
    
    avg_compactness = np.mean(cluster_compactness_values) if cluster_compactness_values else 0.0
    
    return {
        f'dbscan_num_clusters_{thresh_str}': int(num_clusters),
        f'dbscan_avg_inter_cluster_distance_{thresh_str}': float(avg_distance),
        f'dbscan_max_inter_cluster_distance_{thresh_str}': float(max_distance),
        f'dbscan_cluster_compactness_{thresh_str}': float(avg_compactness),
        f'dbscan_noise_ratio_{thresh_str}': float(noise_ratio)
    }


def connected_component_analysis(S, threshold):
    """
    Find connected components in thresholded saliency.
    Fast and interpretable.
    """
    binary_mask = (S >= threshold).cpu().numpy()
    labeled_array, num_components = connected_components(binary_mask)
    
    thresh_str = f"thresh_{threshold:.2f}"
    
    if num_components == 0:
        return {
            f'cc_num_clusters_{thresh_str}': 0,
            f'cc_avg_inter_cluster_distance_{thresh_str}': 0.0,
            f'cc_max_inter_cluster_distance_{thresh_str}': 0.0,
            f'cc_avg_cluster_size_{thresh_str}': 0.0,
            f'cc_cluster_size_ratio_{thresh_str}': 0.0
        }
    
    # Calculate cluster properties
    cluster_centers = []
    cluster_sizes = []
    
    for comp_id in range(1, num_components + 1):
        component_mask = (labeled_array == comp_id)
        coords = np.argwhere(component_mask)
        
        if len(coords) > 0:
            cluster_centers.append(coords.mean(axis=0))
            cluster_sizes.append(len(coords))
    
    cluster_centers = np.array(cluster_centers)
    cluster_sizes = np.array(cluster_sizes)
    
    # Inter-cluster distances
    if len(cluster_centers) > 1:
        distances = pdist(cluster_centers, metric='euclidean')
        avg_distance = distances.mean()
        max_distance = distances.max()
    else:
        avg_distance = 0.0
        max_distance = 0.0
    
    # Cluster size statistics
    avg_size = cluster_sizes.mean()
    size_ratio = cluster_sizes.max() / cluster_sizes.sum() if cluster_sizes.sum() > 0 else 0
    
    return {
        f'cc_num_clusters_{thresh_str}': int(num_components),
        f'cc_avg_inter_cluster_distance_{thresh_str}': float(avg_distance),
        f'cc_max_inter_cluster_distance_{thresh_str}': float(max_distance),
        f'cc_avg_cluster_size_{thresh_str}': float(avg_size),
        f'cc_cluster_size_ratio_{thresh_str}': float(size_ratio),
        f'num_salient_pixels_{thresh_str}': int(cluster_sizes.sum()),
        f'salient_pixels_percent_{thresh_str}': float(cluster_sizes.sum() /  S.numel() * 100)
    }