# visualization/plots.py
"""
Plotting and visualization utilities for saliency analysis.

Provides functions for:
- Drawing mask contours
- Creating metrics tables
- Visualizing saliency maps and analysis results
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib import patches


def draw_mask_contour(ax, mask, color=[0, 0, 0], linewidth=2):
    """
    Draw contour around mask regions instead of overlay.
    
    Args:
        ax: Matplotlib axis
        mask (torch.Tensor or ndarray): Binary mask
        color (list): RGB color for contour [0-1]
        linewidth (int): Line width for contour
    """
    from scipy.ndimage import binary_dilation, binary_erosion
    
    mask_np = mask.cpu().numpy() if hasattr(mask, 'cpu') else mask
    
    # Find edges by dilating and subtracting original
    dilated = binary_dilation(mask_np)
    edges = dilated & ~mask_np
    
    # Draw the edges
    colored = np.zeros((*edges.shape, 3))
    colored[..., :] = color
    ax.imshow(colored, alpha=edges.astype(float))

import re
def extract_seed_from_name(model_name):
    """
    Extract seed number from model name string.
    
    Examples:
        'ResNet, BN=False, Bias=1.0, seed=109' -> 'seed=109'
        '19-11-2025_19-48-17_ResNet, BN=False, Bias=1.0, seed=109...' -> 'seed=109'
    
    Args:
        model_name (str): Model name string
        
    Returns:
        str: Extracted seed or original name if not found
    """
    match = re.search(r'seed=(\d+)', model_name)
    if match:
        return f"seed={match.group(1)}"
    return model_name  # fallback if no seed found

def create_metrics_table(ax, metrics_list, model_names):
    """
    Create a clean transposed table with averaged metrics.
    
    Args:
        ax: Matplotlib axis
        metrics_list (list): List of metric dictionaries
        model_names (list): List of model name strings
    """
    ax.axis('off')
    
    # Compute averages
    avg_metrics = {}
    metric_keys = [ 
                   'coverage_eyes', 'coverage_nose', 'coverage_mouth', 
                   'attribution_eyes', 'attribution_nose', 'attribution_mouth']
    
    for key in metric_keys:
        if key in metrics_list[0]:
            avg_metrics[key] = np.mean([m[key] for m in metrics_list])
    
    # Calculate total coverage and attribution
    total_coverage = (avg_metrics.get('coverage_eyes', 0) + 
                     avg_metrics.get('coverage_nose', 0) + 
                     avg_metrics.get('coverage_mouth', 0))
    
    total_attribution = (avg_metrics.get('attribution_eyes', 0) + 
                        avg_metrics.get('attribution_nose', 0) + 
                        avg_metrics.get('attribution_mouth', 0))
    
    # Format table data - TRANSPOSED (metrics as columns)
    table_data = [
        ['Metric', 'Average'],
        ['Eyes Coverage', f"{avg_metrics.get('coverage_eyes', 0):.1%}"],
        ['Nose Coverage', f"{avg_metrics.get('coverage_nose', 0):.1%}"],
        ['Mouth Coverage', f"{avg_metrics.get('coverage_mouth', 0):.1%}"],
        ['Total Coverage', f"{total_coverage:.1%}"],
        ['Eyes Attribution', f"{avg_metrics.get('attribution_eyes', 0):.1%}"],
        ['Nose Attribution', f"{avg_metrics.get('attribution_nose', 0):.1%}"],
        ['Mouth Attribution', f"{avg_metrics.get('attribution_mouth', 0):.1%}"],
        ['Total Attribution', f"{total_attribution:.1%}"],
    ]
    
    # Transpose: convert rows to columns
    transposed_data = [list(row) for row in zip(*table_data)]
    
    table = ax.table(cellText=transposed_data, 
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.08] * len(table_data))
    
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 2)
    
    # Style: Black & White formal design
    # Header row (first row)
    for i in range(len(table_data)):
        cell = table[(0, i)]
        cell.set_facecolor('black')
        cell.set_text_props(weight='bold', color='white')
        cell.set_edgecolor('black')
        cell.set_linewidth(1.5)
    
    # Data row (second row)
    for i in range(len(table_data)):
        cell = table[(1, i)]
        cell.set_facecolor('white')
        cell.set_text_props(color='black')
        cell.set_edgecolor('black')
        cell.set_linewidth(1)


def visualize_saliency_row(
    image,
    saliency_maps,
    masks,
    metrics,
    model_names,
    save_path=None
):
        """
        Layout:
        Row 1: model1 | model2 | model3 | model4 | model5
        Row 2: metrics table (spanning all columns)
        """
        n = len(saliency_maps)
        
        # Create figure with 2 rows: images on top, table below
        fig = plt.figure(figsize=(4 * n, 5.5))
        
        # Create grid: top row for images, bottom row for table
        # Reduced bottom row height and reduced hspace for smaller margin
        gs = fig.add_gridspec(2, n, height_ratios=[4, 1], hspace=0.15)
        
        # Top row: saliency visualizations
        for i in range(n):
            ax = fig.add_subplot(gs[0, i])
            
            # Display image
            ax.imshow(image.cpu().detach().numpy(), cmap="gray")
            
            # Overlay saliency
            ax.imshow(saliency_maps[i].cpu().detach().numpy(), 
                    alpha=0.75, cmap="hot", vmin=0, vmax=1,
                    interpolation='bicubic')

            # Draw mask contours (black outlines)
            if masks:
                draw_mask_contour(ax, masks["eyes"], color=[0, 0, 0], linewidth=2)
                draw_mask_contour(ax, masks["nose"], color=[0, 0, 0], linewidth=2)
                draw_mask_contour(ax, masks["mouth"], color=[0, 0, 0], linewidth=2)

            # Extract and display only seed number as title
            seed_label = extract_seed_from_name(model_names[i])
            ax.set_title(seed_label, fontsize=10, pad=5)

            ax.axis("off")

        # Bottom row: metrics table spanning all columns
        ax_table = fig.add_subplot(gs[1, :])
        create_metrics_table(ax_table, metrics, model_names)

        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved to {save_path}")
        
        plt.close()  # Close to free memory