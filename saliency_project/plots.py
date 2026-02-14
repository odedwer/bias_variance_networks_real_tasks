# visualization/plots.py

import matplotlib.pyplot as plt
import numpy as np


def overlay_mask(ax, mask, color, alpha=0.4):
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
        ax.imshow(image.cpu().detach().numpy(), cmap="gray")
        ax.imshow(saliency_maps[i].cpu().detach().numpy(), alpha=0.75, cmap="hot", vmin=0, vmax=1,
                               interpolation='bicubic')

        # ax.imshow(image)
        # ax.imshow(saliency_maps[i].cpu(), cmap="hot", alpha=0.6, vmin=0, vmax=vmax)

        overlay_mask(ax, masks["eyes"], color=[1, 1, 1])
        overlay_mask(ax, masks["nose"], color=[1, 1, 1])
        overlay_mask(ax, masks["mouth"], color=[1, 1, 1])

        m = metrics[i]
        ax.set_title(
            f"{model_names[i]}\n"
            f"Entropy={m['entropy']:.2f}, Max Short Distance={m['max_short_distance']:.1f}\n"
            f"Mean Short Distance={m['mean_short_distance']:.1f}, Top 5% Concentration={m['top_5%_concentration']:.2f}\n"
            f"E={m['coverage_eyes']:.2f} N={m['coverage_nose']:.2f} M={m['coverage_mouth']:.2f}\n"
            f"aE={m['attribution_eyes']:.2f} aN={m['attribution_nose']:.2f} aM={m['attribution_mouth']:.2f}"
            ,fontsize=9
        )

        ax.axis("off")

    plt.tight_layout()
    plt.show()