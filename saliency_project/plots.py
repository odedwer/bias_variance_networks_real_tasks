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
    Create a clean transposed table with averaged metrics.
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
    
    # Format table data - TRANSPOSED (metrics as columns)
    table_data = [
        ['Metric', 'Average'],
        ['Entropy', f"{avg_metrics.get('normalized_entropy', 0):.2f}"],
        ['Max Dist', f"{avg_metrics.get('max_short_distance', 0):.1f}"],
        ['Mean Dist', f"{avg_metrics.get('mean_short_distance', 0):.1f}"],
        ['Eyes Cov', f"{avg_metrics.get('coverage_eyes', 0):.1%}"],
        ['Nose Cov', f"{avg_metrics.get('coverage_nose', 0):.1%}"],
        ['Mouth Cov', f"{avg_metrics.get('coverage_mouth', 0):.1%}"],
        ['Total Cov', f"{total_coverage:.1%}"],
        ['Eyes Attr', f"{avg_metrics.get('attribution_eyes', 0):.1%}"],
        ['Nose Attr', f"{avg_metrics.get('attribution_nose', 0):.1%}"],
        ['Mouth Attr', f"{avg_metrics.get('attribution_mouth', 0):.1%}"],
        ['Total Attr', f"{total_attribution:.1%}"],
    ]
    
    # Transpose: convert rows to columns
    transposed_data = [list(row) for row in zip(*table_data)]
    
    table = ax.table(cellText=transposed_data, 
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.08] * len(table_data))
    
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.5)
    
    # Style header column (first row after transpose)
    for i in range(len(table_data)):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate column colors
    for i in range(len(transposed_data)):
        for j in range(len(table_data)):
            if j % 2 == 0 and i > 0:
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
    Layout:
    Row 1: model1 | model2 | model3 | model4 | model5
    Row 2: metrics table (spanning all columns)
    """
    n = len(saliency_maps)
    
    # Create figure with 2 rows: images on top, table below
    fig = plt.figure(figsize=(4 * n, 6))
    
    # Create grid: top row for images, bottom row for table
    gs = fig.add_gridspec(2, n, height_ratios=[4, 1.5], hspace=0.3)
    
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

        # Model name as title - wrap text to fit
        title_text = model_names[i]
        ax.set_title(title_text, fontsize=8, wrap=True)

        ax.axis("off")

    # Bottom row: metrics table spanning all columns
    ax_table = fig.add_subplot(gs[1, :])
    create_metrics_table(ax_table, metrics, model_names)

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.close()  # Close to free memory           ]