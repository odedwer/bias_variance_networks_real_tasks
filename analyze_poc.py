import os
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import numpy as np
from face_recognition_model_comparison import SimpleCNN
from ResNet import ResNet
from poc_bias_variance import FER2013BinaryDataset
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from model_analysis import test_transforms
from torch_cka import CKA
from utils import get_device
import warnings
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
                SimpleCNN(num_classes=2,bn=False, init_bias=float(params.iloc[-1,-1]) if params.iloc[-1,-1]!= 'None' else None)
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
        self.dataloader = DataLoader(self.dataset, batch_size=256, shuffle=False, num_workers=8, persistent_workers=True, pin_memory=True)
        self.device = device
        # calculate accuracy
        self.accuracy = []
        self.model_name = model_chk_path.split("_")[2] if "POC" not in model_chk_path else model_chk_path.split("-")[-1][7:]
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
            saliency_maps.append(self._norm(image.grad.abs()[0]))
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
                os.path.join("figures", "saliency_maps", self.model_name, f"{self.model_name}_img_{img_idx}_class_{label}.pdf")
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

    # <<< START OF FIX >>>
    with torch.no_grad():  # Add this context manager
        with torch.amp.autocast('cuda', enabled=False):
            cka = CKA(
                ma1.models[epoch_idx1],
                ma2.models[epoch_idx2],
                model1_name=ma1.model_name,
                model2_name=ma2.model_name,
                model1_layers=ma_layers1,
                model2_layers=ma_layers2,
                device=ma1.device
            )
            try:
                cka.compare(ma1.dataloader)
            except AssertionError:
                pass
            results = cka.export()
    if plot:
        fig = plt.figure(figsize=(15, 15))
        col = plt.imshow(results['CKA'], vmin=0, vmax=1, cmap="coolwarm", origin="lower")
        plt.colorbar(col)
        # annotate the heatmap
        for i in range(results['CKA'].shape[0]):
            for j in range(results['CKA'].shape[1]):
                plt.text(j, i, f"{results['CKA'][i, j]:.2f}", ha="center", va="center", color="black")

        plt.xticks(range(len(results['model2_layers'])), results['model2_layers'], rotation=90)
        plt.yticks(range(len(results['model1_layers'])), results['model1_layers'])
        plt.title(f"CKA comparison between {ma1.model_name} and {ma2.model_name}", fontsize=18, fontweight="bold")
        plt.xlabel(ma2.model_name+f" epoch {ma2.epochs[epoch_idx2]}", fontsize=15, fontweight="bold")
        plt.ylabel(ma1.model_name+f" epoch {ma2.epochs[epoch_idx2]}", fontsize=15, fontweight="bold")
        plt.tight_layout()
        os.makedirs(os.path.join("figures", "cka"), exist_ok=True)
        if save:
            plt.savefig(
                os.path.join("figures", "cka",
                             f"{ma1.model_name}_{ma2.model_name}_epoch1_{ma1.epochs[epoch_idx1]}_epoch2_{ma2.epochs[epoch_idx2]}.pdf")
            )
        if show:
            plt.show()
        else:
            plt.close('all')
    return results



def run_analysis():
    # !!! IMPORTANT !!!
    # UPDATE bdthese paths with the timestamped folder names created by poc_bias_variance.py
    # Look inside your 'runs' or 'models' directory for folders starting with "POC_"
    LOW_VAR_EXP_DIR = "28-10-2025_12-52-37_POC_Low_Variance_Bias_seed2"
    HIGH_VAR_EXP_DIR = "28-10-2025_12-56-07_POC_High_Variance_Bias_seed2"

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

    # We still need ModelAnalysis to load the models and dataloaders for CKA
    # Note: The 'happy'/'sad' test set is a subset of the full test set, so the dataloader is compatible.
    ma_low_var = ModelAnalysis(LOW_VAR_EXP_DIR, device)
    ma_high_var = ModelAnalysis(HIGH_VAR_EXP_DIR, device)
    cka_progression = {
        "Low Variance Bias":None, "High Variance Bias":None
    }
    for ma, name in zip([ma_low_var, ma_high_var],
                        ["Low Variance Bias", "High Variance Bias"]):

        layer_names = [name for name, module in ma.models[-1].named_modules() if
                       isinstance(module, (torch.nn.Conv2d, torch.nn.Linear))]
        cka_over_epochs = {
            'conv1-conv2':[],
            'conv2-fc1':[],
            'fc1-fc2':[]
        }
        for epoch in tqdm(range(len(ma_low_var.models))):
            res_low = cka_comparison(
                epoch_idx1=epoch,
                ma1=ma,
                ma_layers1=layer_names,
                epoch_idx2=epoch,
                plot=True,
                show=False,
                save=True
            )
            # save cka results for specific layer pairs
            cka_over_epochs['conv1-conv2'].append(res_low['CKA'][0,1].item())
            cka_over_epochs['conv2-fc1'].append(res_low['CKA'][1,2].item())
            cka_over_epochs['fc1-fc2'].append(res_low['CKA'][2,3].item())
            # Free up GPU memory
            torch.cuda.empty_cache()
            del res_low
        cka_progression[name] = cka_over_epochs.copy()
    # Plot CKA over epochs for both models
    epochs = list(range(1, len(ma_low_var.models) + 1))
    plt.figure(figsize=(10, 6))
    for name,cka_over_epochs_model in cka_progression.items():
        for layer_pair in cka_over_epochs_model.keys():
            plt.plot(epochs, cka_over_epochs_model[layer_pair], label=f'{name} - {layer_pair}', linestyle='--' if 'High' in name else '-')
    plt.xlabel('Epochs')
    plt.ylabel('CKA Similarity')
    plt.title('CKA Similarity Over Epochs for Low and High Variance Models')
    plt.legend()
    os.makedirs("figures/cka_over_epochs", exist_ok=True)
    plt.savefig("figures/cka_over_epochs/cka_over_epochs_comparison.pdf")
    plt.close('all')



if __name__ == '__main__':
    run_analysis()
