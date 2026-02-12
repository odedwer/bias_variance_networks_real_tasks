# visualization/plots.py

import matplotlib.pyplot as plt
import numpy as np


def overlay_mask(ax, mask, color, alpha=0.3):
    colored = np.zeros((*mask.shape, 3))
    colored[..., :] = color
    ax.imshow(colored, alpha=alpha * mask)


def visualize_saliency_row(
    image,
    saliency_maps,
    masks,
    metrics,
    model_names
):
    """
    One row:
    image | model1 | model2 | model3 | model4 | model5
    """
    n = len(saliency_maps)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))

    vmax = max(s.max().item() for s in saliency_maps)

    for i, ax in enumerate(axes):
        ax.imshow(image)
        ax.imshow(saliency_maps[i], cmap="hot", alpha=0.6, vmin=0, vmax=vmax)

        overlay_mask(ax, masks["eyes"], color=[0, 0, 1])
        overlay_mask(ax, masks["nose"], color=[0, 1, 0])
        overlay_mask(ax, masks["mouth"], color=[1, 0, 0])

        m = metrics[i]
        ax.set_title(
            f"{model_names[i]}\n"
            f"H={m['entropy']:.2f}, D={m['max_short_distance']:.1f}\n"
            f"E={m['eyes']:.2f} N={m['nose']:.2f} M={m['mouth']:.2f}",
            fontsize=9
        )

        ax.axis("off")

    plt.tight_layout()
    plt.show()