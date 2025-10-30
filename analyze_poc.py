import os
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import numpy as np
import hashlib  # <-- Added for fingerprinting
import json  # <-- Added for fingerprinting
from face_recognition_model_comparison import SimpleCNN
from ResNet import ResNet
from poc_bias_variance import FER2013BinaryDataset
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from model_analysis import test_transforms
from torch_cka import CKA
from utils import get_device
import warnings


# --- (ModelAnalysis class remains unchanged) ---
class ModelAnalysis:
    def __init__(self, model_chk_path, device):
        self.epochs = []
        model_chkpoints = os.listdir(os.path.join("models", model_chk_path))
        self.params = pd.read_csv(os.path.join("runs", model_chk_path, "params.csv"))
        if "SimpleCNN" in model_chk_path:
            self.models = [
                SimpleCNN(bn="BN=True" in model_chk_path, init_bias=0.0 if "Bias=None" not in model_chk_path else None)
                for _ in range(len(model_chkpoints))]
        elif "ResNet" in model_chk_path:
            self.models = [ResNet(bn="BN=True" in model_chk_path, bias="Bias=None" not in model_chk_path) for _ in
                           range(len(model_chkpoints))]
        elif "POC" in model_chk_path:
            params = pd.read_csv(os.path.join("runs", model_chk_path, "params.csv"))

            self.models = [
                SimpleCNN(num_classes=2, bn=False,
                          init_bias=float(params.iloc[-1, -1]) if params.iloc[-1, -1] != 'None' else None)
                for _ in range(len(model_chkpoints))
            ]

        # sort by creation date
        model_chkpoints.sort(key=lambda x: os.path.getctime(os.path.join("models", model_chk_path, x)))

        for i, filename in enumerate(model_chkpoints):
            if i == 0:
                self.epochs.append("Init")
            elif "final" in filename:
                self.epochs.append("after training")
            else:
                self.epochs.append(int(filename.split("-")[1].split(".")[0]))
            self.models[i].load_state_dict(torch.load(os.path.join("models", model_chk_path, filename)))
            self.models[i] = self.models[i].to(device)
        for model in self.models:
            model.eval()
        self.dataset = FER2013BinaryDataset('data/face-expression/test', transform=test_transforms)
        self.dataloader = DataLoader(self.dataset, batch_size=128, shuffle=False, num_workers=4,
                                     persistent_workers=True, pin_memory=True)
        self.device = device
        # calculate accuracy
        self.accuracy = []
        self.model_name = model_chk_path.split("_")[2] if "POC" not in model_chk_path else model_chk_path.split("-")[
                                                                                               -1][7:]
        for model in self.models:
            correct = 0
            total = 0
            with torch.no_grad():
                for images, labels in self.dataloader:
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            self.accuracy.append(100 * correct / total)

    @staticmethod
    def _norm(x):
        return (x - x.min()) / (x.max() - x.min())

    def visualize_filters(self, show=True, save=False):
        models = [self.models[0], self.models[-1]]
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        for model, ax in zip(models, axes):
            for name, param in model.named_parameters():
                if "conv" in name:
                    # scale parameters to 0-255 range
                    param = 255 * self._norm(param)
                    ax.imshow(make_grid(param).cpu().detach().numpy().astype(int).transpose((1, 2, 0)))
                    break
        axes[0].set(title="Initialization")
        axes[1].set(title="After training")
        if save:
            os.makedirs(os.path.join("figures", "filters"), exist_ok=True)
            plt.savefig(os.path.join("figures", "filters", f"{self.model_name}_filters.pdf"))
        if show:
            plt.show()

    def visualize_saliency_map(self, img_idx, show=True, save=False):
        image, label = self.dataset[img_idx]
        image = image.to(self.device)
        image.requires_grad = True
        label = self.dataset.number_label_map[label]
        saliency_maps = []
        for model in self.models:
            model.zero_grad()
            image.requires_grad = True
            output = model(image[None, ...].to(self.device))
            output[0, output.argmax()].backward()

            # --- START OF FIX for potential zero gradient ---
            saliency = image.grad.abs()[0]
            s_min = saliency.min()
            s_max = saliency.max()
            s_range = s_max - s_min
            # Add a small epsilon to prevent division by zero
            norm_saliency = (saliency - s_min) / (s_range + 1e-8)
            saliency_maps.append(norm_saliency)
            # --- END OF FIX ---

            # plot the saliency map on top of the image
        n_rows, n_cols = np.sqrt(len(saliency_maps)).astype(int), np.ceil(
            len(saliency_maps) / np.sqrt(len(saliency_maps))).astype(int)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 10))
        try:
            axes = axes.flatten()
        except AttributeError:
            axes = [axes]
        # share all heatmap colorbars
        for i, saliency_map in enumerate(saliency_maps):
            axes[i].imshow(image[0].cpu().detach().numpy(), cmap="gray")
            # calculate smooth
            c = axes[i].imshow(saliency_map.cpu().detach().numpy(), alpha=0.5, cmap="hot", vmin=0, vmax=1,
                               interpolation='bicubic')

            axes[i].set_title("Epoch: " + str(self.epochs[i]) + f", Acc:{self.accuracy[i]:.2f}" + ", : " + (
                self.dataset.number_label_map[output.argmax().item()]))
            fig.colorbar(c, ax=axes[i], orientation="horizontal")
            # remove axis

        for i in range(0, len(axes)):
            axes[i].axis("off")
        fig.suptitle(f"{self.model_name} Saliency map, img {img_idx}, class {label}")
        fig.tight_layout()
        if save:
            os.makedirs(os.path.join("figures", "saliency_maps", self.model_name), exist_ok=True)
            plt.savefig(
                os.path.join("figures", "saliency_maps", self.model_name,
                             f"{self.model_name}_img_{img_idx}_class_{label}.pdf")
            )
        if show:
            plt.show()
        else:
            plt.close('all')

    def get_model_layer_names(self, epoch_idx):
        return [l[0] for l in list(self.models[epoch_idx].named_modules()) if l[0] and (
                "conv" in l[0] or "fc" in l[0] or "relu" in l[0])]


def cka_comparison(epoch_idx1: int, ma1: ModelAnalysis, ma_layers1: list[str], epoch_idx2, ma2: ModelAnalysis = None,
                   ma_layers2: list[str] = None, plot=True, show=False, save=False):
    warnings.filterwarnings("ignore", category=UserWarning)
    if ma2 is None:
        ma2 = ma1
    if ma_layers2 is None:
        ma_layers2 = ma_layers1

    # --- Caching Logic ---
    CKA_CACHE_DIR = "results/cka"
    os.makedirs(CKA_CACHE_DIR, exist_ok=True)

    # Create a unique fingerprint for the comparison
    model1_name = ma1.model_name
    model2_name = ma2.model_name
    epoch1_str = str(ma1.epochs[epoch_idx1])
    epoch2_str = str(ma2.epochs[epoch_idx2])

    # Hash the layer lists to create a short, unique ID
    layers1_hash = hashlib.md5(json.dumps(ma_layers1, sort_keys=True).encode()).hexdigest()[:8]
    layers2_hash = hashlib.md5(json.dumps(ma_layers2, sort_keys=True).encode()).hexdigest()[:8]

    filename = f"{model1_name}__{epoch1_str}__{layers1_hash}__VS__{model2_name}__{epoch2_str}__{layers2_hash}.pt"
    cache_path = os.path.join(CKA_CACHE_DIR, filename)

    # Check if results are already cached
    if os.path.exists(cache_path):
        # print(f"Loading cached CKA results from: {cache_path}")
        results = torch.load(cache_path)
    else:
        # print(f"Calculating CKA, saving to: {cache_path}")
        with torch.no_grad():
            with torch.amp.autocast('cuda', enabled=False):
                cka = CKA(
                    ma1.models[epoch_idx1],
                    ma2.models[epoch_idx2],
                    model1_name=model1_name,
                    model2_name=model2_name,
                    model1_layers=ma_layers1,
                    model2_layers=ma_layers2,
                    device=ma1.device
                )
                try:
                    cka.compare(ma1.dataloader)
                except AssertionError:
                    pass
                results = cka.export()

                # Save results to cache
                torch.save(results, cache_path)
                del cka

    # --- Plotting Logic ---
    if plot:
        fig = plt.figure(figsize=(15, 15))
        col = plt.imshow(results['CKA'], vmin=0, vmax=1, cmap="coolwarm", origin="lower")
        plt.colorbar(col)
        # annotate the heatmap
        for i in range(results['CKA'].shape[0]):
            for j in range(results['CKA'].shape[1]):
                plt.text(j, i, f"{results['CKA'][i, j]:.2f}", ha="center", va="center", color="black", fontsize=16, fontweight='bold')

        plt.xticks(range(len(results['model2_layers'])), results['model2_layers'], rotation=90)
        plt.yticks(range(len(results['model1_layers'])), results['model1_layers'])
        plt.title(f"CKA comparison between {ma1.model_name} and {ma2.model_name}", fontsize=18, fontweight="bold")
        plt.xlabel(ma2.model_name + f" epoch {ma2.epochs[epoch_idx2]}", fontsize=15, fontweight="bold")
        plt.ylabel(ma1.model_name + f" epoch {ma1.epochs[epoch_idx1]}", fontsize=15,
                   fontweight="bold")  # Fixed: was ma2.epochs
        plt.tight_layout()

        # Create a unique path for the figure
        FIGURE_CACHE_DIR = "figures/cka"
        os.makedirs(FIGURE_CACHE_DIR, exist_ok=True)
        # Use the same filename but with .pdf
        figure_path = os.path.join(FIGURE_CACHE_DIR, os.path.splitext(filename)[0] + ".svg")

        if save:
            plt.savefig(figure_path)
        if show:
            plt.show()

        plt.close(fig)  # Close the specific figure

    return results


def run_analysis():
    # !!! IMPORTANT !!!
    # UPDATE these paths with the timestamped folder names created by poc_bias_variance.py
    # Look inside your 'runs' or 'models' directory for folders starting with "POC_"
    # Find the seed 1, 2, or 3 folders
    LOW_VAR_EXP_DIR = "28-10-2025_20-02-21_POC_Low_Variance_Bias_seed2"
    HIGH_VAR_EXP_DIR = "28-10-2025_20-10-23_POC_High_Variance_Bias_seed2"

    device = get_device()

    # --- 1. Performance Comparison ---
    print("\n--- Performance Comparison ---")
    print("✅ The definitive test accuracy for each model was printed at the end of the `poc_bias_variance.py` script.")
    print("Please refer to that terminal output for the final performance numbers.")

    # --- 2. CKA Comparison ---
    print("\nRunning CKA comparison between the final models...")
    if not os.path.exists(os.path.join("models", LOW_VAR_EXP_DIR)):
        raise FileNotFoundError(f"Experiment directory not found: {LOW_VAR_EXP_DIR}. Please check the path.")
    if not os.path.exists(os.path.join("models", HIGH_VAR_EXP_DIR)):
        raise FileNotFoundError(f"Experiment directory not found: {HIGH_VAR_EXP_DIR}. Please check the path.")

    ma_low_var = ModelAnalysis(LOW_VAR_EXP_DIR, device)
    ma_high_var = ModelAnalysis(HIGH_VAR_EXP_DIR, device)
    cka_progression = {
        ma_low_var.model_name: None,
        ma_high_var.model_name: None
    }

    # --- Caching for CKA progression ---
    CKA_PROGRESSION_CACHE_DIR = "results/cka_over_epochs"
    os.makedirs(CKA_PROGRESSION_CACHE_DIR, exist_ok=True)

    print("Analyzing CKA progression over epochs...")
    for ma, name in zip([ma_low_var, ma_high_var],
                        [ma_low_var.model_name, ma_high_var.model_name]):  # Use dynamic names

        # --- Check if progression data is already cached ---
        progression_cache_path = os.path.join(CKA_PROGRESSION_CACHE_DIR, f"{name}_cka_progression.pt")
        # if os.path.exists(progression_cache_path):
        #     print(f"Loading cached CKA progression for {name} from: {progression_cache_path}")
        #     cka_progression[name] = torch.load(progression_cache_path)
        #     continue  # Skip to the next model
        # --- End cache check ---

        layer_names = [name for name, module in ma.models[-1].named_modules() if
                       isinstance(module, (torch.nn.Conv2d, torch.nn.Linear))]
        cka_over_epochs = {
            'conv1-conv2': [],
            'conv2-fc1': [],
            'fc1-fc2': []
        }

        # Use the correct model length for the loop
        for epoch in tqdm(range(len(ma.models)), desc=f"Analyzing {ma.model_name}"):
            res = cka_comparison(
                epoch_idx1=epoch,
                ma1=ma,
                ma_layers1=layer_names,
                epoch_idx2=epoch,  # Compare model to itself at this epoch
                ma2=ma,
                ma_layers2=layer_names,
                plot=True,
                show=False,
                save=True
            )
            # save cka results for specific layer pairs
            cka_over_epochs['conv1-conv2'].append(res['CKA'][0, 1].item())
            cka_over_epochs['conv2-fc1'].append(res['CKA'][1, 2].item())
            cka_over_epochs['fc1-fc2'].append(res['CKA'][2, 3].item())

            # Free up GPU memory
            torch.cuda.empty_cache()
            del res

        cka_progression[name] = cka_over_epochs.copy()

        # --- Save the computed progression data ---
        print(f"Saving CKA progression for {name} to: {progression_cache_path}")
        torch.save(cka_progression[name], progression_cache_path)

    # Plot CKA over epochs for both models
    print("Plotting CKA progression...")
    epochs = ma_low_var.epochs  # Use the epoch labels from ModelAnalysis

    plt.figure(figsize=(10, 6))
    for name, cka_over_epochs_model in cka_progression.items():
        # reset plot color cycle
        plt.gca().set_prop_cycle(None)
        if cka_over_epochs_model is None:  # Handle case where a model might not have run
            continue

        # Ensure epoch list and data list match in length
        data_len = len(cka_over_epochs_model['conv1-conv2'])
        epoch_labels = np.arange(1,data_len+1,5)
        epoch_ticks = epoch_labels

        for layer_pair in cka_over_epochs_model.keys():
            data = cka_over_epochs_model[layer_pair][::5]
            plt.plot(epoch_ticks[1:], data[1:], label=f'{name} - {layer_pair}', linestyle='--' if 'High' in name else '-')

    # Set x-ticks to be the epoch labels
    plt.xticks(ticks=epoch_ticks, labels=epoch_labels,fontsize=9)
    plt.xlabel('Epochs')
    plt.ylabel('CKA Similarity')
    plt.title('CKA Similarity Over Epochs for Low and High Variance Models')
    # remove top and right spines
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.legend(loc='upper right')
    plt.tight_layout()

    os.makedirs(CKA_PROGRESSION_CACHE_DIR, exist_ok=True)
    plt.savefig(os.path.join(CKA_PROGRESSION_CACHE_DIR, "cka_over_epochs_comparison.pdf"))
    plt.close('all')
    print("✅ CKA analysis complete. Plots saved to 'figures/cka' and 'results/cka_over_epochs'.")


if __name__ == '__main__':
    run_analysis()

