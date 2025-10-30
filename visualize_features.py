import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from torchvision.utils import make_grid
import warnings

# --- Imports from your existing files ---
# We need the model definition (SimpleCNN) and the device util
from face_recognition_model_comparison import SimpleCNN
from utils import get_device
# Use the same transformations as your test set
from model_analysis import test_transforms

# --- CONFIGURATION ---

# !!! IMPORTANT !!!
# You MUST edit these lists to include the filenames of the images you want to analyze.
# The paths are relative to where you run the script.
#
# These are placeholder names.
# Example: "data/face-expression/test/happy/im100.png"
HAPPY_IMAGES = [
    "data/face-expression/test/happy/PrivateTest_12052491.jpg",
    "data/face-expression/test/happy/PrivateTest_10077120.jpg",
    "data/face-expression/test/happy/PrivateTest_9668583.jpg",
    "data/face-expression/test/happy/PrivateTest_7018994.jpg",
    "data/face-expression/test/happy/PrivateTest_3579299.jpg",
    "data/face-expression/test/happy/PrivateTest_2028370.jpg",
    "data/face-expression/test/happy/PrivateTest_1735299.jpg",
    "data/face-expression/test/happy/PrivateTest_928647.jpg",
    "data/face-expression/test/happy/PrivateTest_218533.jpg",
    "data/face-expression/test/happy/PrivateTest_95094.jpg",
]

# Example: "data/face-expression/test/sad/im10.png"
SAD_IMAGES = [
    "data/face-expression/test/sad/PrivateTest_1414350.jpg",
    "data/face-expression/test/sad/PrivateTest_2892457.jpg",
    "data/face-expression/test/sad/PrivateTest_5594100.jpg",
    "data/face-expression/test/sad/PrivateTest_6389977.jpg",
    "data/face-expression/test/sad/PrivateTest_7577600.jpg",
    "data/face-expression/test/sad/PrivateTest_9859012.jpg",
    "data/face-expression/test/sad/PrivateTest_9966612.jpg",
    "data/face-expression/test/sad/PrivateTest_11825918.jpg",
    "data/face-expression/test/sad/PrivateTest_21188058.jpg",
    "data/face-expression/test/sad/PrivateTest_29476048.jpg",
]

# These labels correspond to the FER2013BinaryDataset class mapping
# 0: happy, 1: sad
CLASS_MAP = {
    "happy": 0,
    "sad": 1
}

OUTPUT_DIR = "figures/feature_viz"


# --- END CONFIGURATION ---


def load_models(device):
    """
    Finds all 'Low' and 'High' variance POC models and loads their
    final trained state.
    """
    low_var_models = []
    high_var_models = []

    models_dir = "models"
    runs_dir = "runs"

    if not os.path.exists(models_dir):
        print(f"Error: 'models' directory not found.")
        return [], []
    if not os.path.exists(runs_dir):
        print(f"Error: 'runs' directory not found.")
        return [], []

    print(f"Scanning {models_dir} for POC models...")
    for model_dir in os.listdir(models_dir):
        model_path = os.path.join(models_dir, model_dir)
        if not os.path.isdir(model_path) or "POC" not in model_dir:
            continue

        # 1. Find the final trained model
        final_model_path = os.path.join(model_path, "epoch-final.pth")
        if not os.path.exists(final_model_path):
            continue

        # 2. Find the params to get the init_bias
        params_path = os.path.join(runs_dir, model_dir, "params.csv")
        if not os.path.exists(params_path):
            print(f"Warning: Could not find params.csv for {model_dir}, skipping.")
            continue

        try:
            # Re-using the logic from your analyze_poc.py to load bias
            params = pd.read_csv(params_path)
            bias_val = params.iloc[-1, -1]
            init_bias = float(bias_val) if bias_val != 'None' else None

            # 3. Load the model
            model = SimpleCNN(num_classes=2, bn=False, init_bias=init_bias).to(device)
            model.load_state_dict(torch.load(final_model_path, map_location=device))
            model.eval()

            model_data = {
                'model': model,
                'name': model_dir,
                'bias': init_bias
            }

            if "Low_Variance_Bias" in model_dir:
                low_var_models.append(model_data)
            elif "High_Variance_Bias" in model_dir:
                high_var_models.append(model_data)

        except Exception as e:
            print(f"Error loading model {model_dir}: {e}")

    return low_var_models, high_var_models


def visualize_filters(low_models, high_models):
    """
    Plots the first layer (conv1) filters for low and high variance models
    in a side-by-side comparison.
    """
    print("Visualizing conv1 filters...")

    num_low = len(low_models)
    num_high = len(high_models)
    num_rows = max(num_low, num_high, 1)  # Ensure at least 1 row

    fig, axes = plt.subplots(num_rows, 2, figsize=(10, 4 * num_rows), squeeze=False)

    def plot_kernels(ax, model, title):
        # Get conv1 weights
        kernels = model.conv1.weight.detach().cpu()
        # Normalize for visualization
        kernels = (kernels - kernels.min()) / (kernels.max() - kernels.min())
        # Plot all 32 filters in a grid (4 rows of 8)
        grid = make_grid(kernels, nrow=8, padding=1)
        ax.imshow(grid.permute(1, 2, 0))
        ax.set_title(title, fontsize=8)
        ax.axis('off')

    # Set main titles for columns
    if axes.shape[0] > 0:
        axes[0, 0].set_title(f"Low Variance Models ({num_low} found)", fontsize=12, pad=20)
        axes[0, 1].set_title(f"High Variance Models ({num_high} found)", fontsize=12, pad=20)

    for i in range(num_rows):
        if i < num_low:
            model_data = low_models[i]
            title = f"Bias: {model_data['bias']}\n{model_data['name']}"
            plot_kernels(axes[i, 0], model_data['model'], title)
        else:
            axes[i, 0].axis('off')  # Hide unused plots

        if i < num_high:
            model_data = high_models[i]
            title = f"Bias: {model_data['bias']}\n{model_data['name']}"
            plot_kernels(axes[i, 1], model_data['model'], title)
        else:
            axes[i, 1].axis('off')  # Hide unused plots

    fig.suptitle("First Layer (conv1) Filters Comparison", fontsize=16, y=1.02)
    fig.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "conv1_filters_comparison.pdf"))
    plt.close(fig)
    print(f"  > Filter comparison saved to {OUTPUT_DIR}/conv1_filters_comparison.pdf")


def load_image(image_path, transform):
    """Loads a single image and prepares it for model input and plotting."""
    try:
        # Load as grayscale ('L') to match dataset
        image = Image.open(image_path).convert('L')
    except FileNotFoundError:
        print(f"    ! Warning: Image not found, skipping: {image_path}")
        return None, None
    except Exception as e:
        print(f"    ! Error loading image {image_path}: {e}")
        return None, None

    # Create normalized tensor for the model
    norm_tensor = transform(image)

    # Create un-normalized tensor for plotting
    plot_tensor = transforms.ToTensor()(image)

    return norm_tensor, plot_tensor


def get_saliency_map(model, image_tensor, target_label, device):
    """
    Gets a saliency map for a single image, IF the model classifies it correctly.
    Returns None if classification is wrong.
    """
    model.zero_grad()
    # Add batch dimension and send to device
    image_tensor = image_tensor.clone().unsqueeze(0).to(device)
    image_tensor.requires_grad = True

    # Forward pass
    output = model(image_tensor)
    _, predicted = torch.max(output.data, 1)

    # --- Check for correct classification ---
    if predicted.item() != target_label:
        # print(f"  ...Skipping image, model predicted {predicted.item()} (expected {target_label})")
        return None

    # Backward pass on the correct target class score
    target_score = output[0, target_label]
    target_score.backward()

    # Get gradients
    saliency = image_tensor.grad.abs().squeeze().cpu()

    # Normalize
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min())
    return saliency.numpy()


def get_average_saliency(model, image_paths, class_label, device, transform):
    """
    Generates an average saliency map from a list of images,
    filtering out incorrect classifications.
    """
    all_maps = []
    for img_path in image_paths:
        norm_tensor, _ = load_image(img_path, transform)
        if norm_tensor is None:
            continue

        saliency_map = get_saliency_map(model, norm_tensor, class_label, device)

        if saliency_map is not None:
            all_maps.append(saliency_map)

    if not all_maps:
        print("    ! Warning: No images were correctly classified for this model. Returning blank map.")
        return np.zeros((48, 48)), 0

    # Return the average map and the count of included images
    return np.nanmean(all_maps, axis=0), len(all_maps)


def plot_saliency(ax, base_image, saliency_map, title):
    """Helper to plot saliency map over a base image."""
    ax.imshow(base_image, cmap='gray')
    ax.imshow(saliency_map, cmap='hot', alpha=0.6)  # Use alpha for blending
    ax.set_title(title, fontsize=8)
    ax.axis('off')


def visualize_saliency(low_models, high_models, image_paths, class_name, class_label, device, transform):
    """
    Generates and plots the side-by-side average saliency maps.
    """
    print(f"Visualizing average saliency for class '{class_name}'...")

    num_low = len(low_models)
    num_high = len(high_models)
    num_rows = max(num_low, num_high, 1)

    fig, axes = plt.subplots(num_rows, 2, figsize=(10, 4 * num_rows), squeeze=False)

    # Load the base image (first in list) to plot on
    _, base_image_tensor = load_image(image_paths[0], transform)
    if base_image_tensor is None:
        print(f"Error: Cannot load base image {image_paths[0]}. Aborting saliency plot.")
        plt.close(fig)
        return

    base_image_np = base_image_tensor.squeeze().cpu().numpy()

    # Set main titles for columns
    if axes.shape[0] > 0:
        axes[0, 0].set_title(f"Low Variance Models ({num_low} found)", fontsize=12, pad=20)
        axes[0, 1].set_title(f"High Variance Models ({num_high} found)", fontsize=12, pad=20)

    for i in range(num_rows):
        if i < num_low:
            model_data = low_models[i]
            avg_map, count = get_average_saliency(model_data['model'], image_paths, class_label, device, transform)
            title = f"Bias: {model_data['bias']} ({count}/{len(image_paths)} correct)\n{model_data['name']}"
            plot_saliency(axes[i, 0], base_image_np, avg_map, title)
        else:
            axes[i, 0].axis('off')

        if i < num_high:
            model_data = high_models[i]
            avg_map, count = get_average_saliency(model_data['model'], image_paths, class_label, device, transform)
            title = f"Bias: {model_data['bias']} ({count}/{len(image_paths)} correct)\n{model_data['name']}"
            plot_saliency(axes[i, 1], base_image_np, avg_map, title)
        else:
            axes[i, 1].axis('off')

    fig.suptitle(f"Average Saliency Map for '{class_name}' (Overlay on {os.path.basename(image_paths[0])})",
                 fontsize=16, y=1.02)
    fig.tight_layout()

    output_filename = os.path.join(OUTPUT_DIR, f"saliency_avg_{class_name}.pdf")
    plt.savefig(output_filename)
    plt.close(fig)
    print(f"  > Saliency comparison for {class_name} saved to {output_filename}")


def main():
    # Suppress warnings
    warnings.filterwarnings("ignore", category=UserWarning)

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = get_device()

    # 1. Load Models
    low_models, high_models = load_models(device)

    if not low_models and not high_models:
        print(
            "No 'POC_Low_Variance_Bias' or 'POC_High_Variance_Bias' models with 'epoch-final.pth' found in 'models/'.")
        print("Please run poc_bias_variance.py first.")
        return

    print(f"Found {len(low_models)} low-variance models.")
    print(f"Found {len(high_models)} high-variance models.")
    print("-" * 30)

    # 2. Visualize Filters
    visualize_filters(low_models, high_models)
    print("-" * 30)

    # 3. Visualize Saliency Maps
    # Check if the placeholder images were modified
    if HAPPY_IMAGES[0] == "data/face-expression/test/happy/im100.png" or \
            SAD_IMAGES[0] == "data/face-expression/test/sad/im10.png":
        print("! WARNING: You are using the default placeholder image lists.")
        print("! Please edit HAPPY_IMAGES and SAD_IMAGES in this script to point to real images.")
        print("! Attempting to run with placeholders anyway...")

    visualize_saliency(low_models, high_models, HAPPY_IMAGES, "happy", CLASS_MAP["happy"], device, test_transforms)
    print("-" * 30)

    visualize_saliency(low_models, high_models, SAD_IMAGES, "sad", CLASS_MAP["sad"], device, test_transforms)
    print("-" * 30)

    print("Feature visualization complete.")


if __name__ == "__main__":
    main()

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# read all CSV files in `accuracies` directory
import os
import glob
# increase plot font size for title (16), axis labels (14) and ticks (13) and legend (12)
plt.rc('axes', titlesize=16)     # Title font size
plt.rc('axes', labelsize=14)    # x and y labels font size
plt.rc('xtick', labelsize=13)    # x tick labels font size
plt.rc('ytick', labelsize=13)    # y tick labels font size
plt.rc('legend', fontsize=12)    # legend font size
# set bold for title and labels
plt.rc('axes', titleweight='bold')
plt.rc('axes', labelweight='bold')
# remove right and top spines
sns.set_style("whitegrid", {'axes.spines.right': False, 'axes.spines.top': False, 'axes.grid': False})
accuracy_files = glob.glob("accuracies/*.csv")
all_accuracies = []
for file in accuracy_files:
    df = pd.read_csv(file)
    df['source_file'] = os.path.basename(file)
    df['model_type'] = df['source_file'].apply(lambda x: 'Low Variance' if 'Low_Variance_Bias' in x else (
        'High Variance' if 'High_Variance_Bias' in x else 'Low Variance'))
    # rename Step to Epoch and Value to accuracy
    df = df.rename(columns={'Step': 'Step', 'Value': 'Accuracy'})
    all_accuracies.append(df)

# concatenate all dataframes
df_all = pd.concat(all_accuracies, ignore_index=True)

# plot accuracy distributions for low and high variance models over epochs
plt.figure(figsize=(10, 6))
sns.lineplot(data=df_all, x='Step', y='Accuracy', hue='model_type', errorbar='sd', palette={
    'Low Variance': "#00A08A",
    'High Variance': '#FF0000'
})
plt.title('Model Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
# add horizontal line at y=0.6, which is chance level for 2-class classification with this imbalanced dataset
plt.axhline(60, color='red', linestyle='--', label='Chance Level (60%)')
plt.legend()
plt.ylim(30, 100)
# save as svg
plt.savefig('figures/accuracy_over_epochs.svg', format='svg')
plt.show()
