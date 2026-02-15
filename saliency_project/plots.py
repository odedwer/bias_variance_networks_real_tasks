# visualization/plots.py

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib import patches


def draw_mask_contour(ax, mask, color=[0, 0, 0], linewidth=2):
    """
    Draw contour around mask regions instead of overlay.
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


def create_metrics_table(ax, metrics_list, model_names):
    """
    Create a clean table with averaged metrics.
    """
    ax.axis('off')
    
    # Compute averages
    avg_metrics = {}
    metric_keys = ['normalized_entropy', 'max_short_distance', 'mean_short_distance', 
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
    
    # Format table data
    table_data = [
        ['Metric', 'Average'],
        ['Entropy', f"{avg_metrics.get('normalized_entropy', 0):.2f}"],
        ['Max Distance', f"{avg_metrics.get('max_short_distance', 0):.1f}"],
        ['Mean Distance', f"{avg_metrics.get('mean_short_distance', 0):.1f}"],
        ['Eyes Coverage', f"{avg_metrics.get('coverage_eyes', 0):.1%}"],
        ['Nose Coverage', f"{avg_metrics.get('coverage_nose', 0):.1%}"],
        ['Mouth Coverage', f"{avg_metrics.get('coverage_mouth', 0):.1%}"],
        ['Total Coverage', f"{total_coverage:.1%}"],
        ['Eyes Attr.', f"{avg_metrics.get('attribution_eyes', 0):.1%}"],
        ['Nose Attr.', f"{avg_metrics.get('attribution_nose', 0):.1%}"],
        ['Mouth Attr.', f"{avg_metrics.get('attribution_mouth', 0):.1%}"],
        ['Total Attr.', f"{total_attribution:.1%}"],
    ]
    
    table = ax.table(cellText=table_data, 
                     cellLoc='left',
                     loc='center',
                     colWidths=[0.6, 0.4])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style header row
    for i in range(2):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(table_data)):
        for j in range(2):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')


def visualize_saliency_row(
    image,
    saliency_maps,
    masks,
    metrics,
    model_names,
    save_path=None
):
    """
    One row:
    metrics table | model1 | model2 | model3 | model4 | model5
    """
    n = len(saliency_maps)
    fig, axes = plt.subplots(1, n + 1, figsize=(4 * (n + 1), 4))

    # First subplot: metrics table
    create_metrics_table(axes[0], metrics, model_names)

    # Remaining subplots: saliency visualizations
    for i in range(n):
        ax = axes[i + 1]
        
        # Display image
        ax.imshow(image.cpu().detach().numpy(), cmap="gray")
        
        # Overlay saliency
        ax.imshow(saliency_maps[i].cpu().detach().numpy(), 
                 alpha=0.75, cmap="hot", vmin=0, vmax=1,
                 interpolation='bicubic')

        # Draw mask contours (black outlines)
        if masks:
            draw_mask_contour(ax, masks["eyes"], color=[0, 0, 0], linewidth=1)
            draw_mask_contour(ax, masks["nose"], color=[0, 0, 0], linewidth=1)
            draw_mask_contour(ax, masks["mouth"], color=[0, 0, 0], linewidth=1)

        # Simple title with just model name
        ax.set_title(model_names[i], fontsize=10)

        ax.axis("off")

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.close()  # Close to free memory