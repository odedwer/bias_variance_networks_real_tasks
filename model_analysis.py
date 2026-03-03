# %%
from itertools import combinations, combinations_with_replacement

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from torch_cka import CKA
import os

from tqdm import tqdm

from utils import get_device
from collections import defaultdict
from ResNet import ResNet
from face_recognition_model_comparison import SimpleCNN, test_transforms, FER2013Dataset

MODELS_FOLDER_PATH = "models/SimpleCNN/bias=10.0"
OUTPUT_CSV = "cka_model_analysis_simplecnn10.csv"
classes=["fear","angry"]
 
class ModelAnalysis:
    """
    a model after training different epochs
    """
    def __init__(self, model_chk_path, device):
        self.epochs_names = []
        model_chkpoints = os.listdir(os.path.join(MODELS_FOLDER_PATH, model_chk_path))
        #self.params = pd.read_csv(os.path.join("runs", model_chk_path, "params.csv"))
        #create appropriate model instances
        if "SimpleCNN" in model_chk_path:
            self.epochs = [
                SimpleCNN(bn="BN=True" in model_chk_path, init_bias=0.0 if "Bias=None" not in model_chk_path else None)
                for _ in range(len(model_chkpoints))]
        elif "ResNet" in model_chk_path:
            self.epochs = [ResNet(bn="BN=True" in model_chk_path, bias="Bias=None" not in model_chk_path) for _ in
                           range(len(model_chkpoints))]
        # sort by creation date - all of the model's checkpoints
        model_chkpoints.sort(key=lambda x: os.path.getctime(os.path.join(MODELS_FOLDER_PATH, model_chk_path, x)))

        for i, filename in enumerate(model_chkpoints):
            if i == 0:
                self.epochs_names.append("Init")
            elif "final" in filename:
                self.epochs_names.append("after training")
            else:
                self.epochs_names.append(int(filename.split("-")[1].split(".")[0]))
            # load model parameters into the created model instances
            self.epochs[i].load_state_dict(torch.load(os.path.join(MODELS_FOLDER_PATH, model_chk_path, filename)))
            self.epochs[i] = self.epochs[i].to(device)
        #set models to evaluation mode
        for model in self.epochs:
            model.eval()

        self.dataset = FER2013Dataset('data/face-expression/test', transform=test_transforms, classes=classes)
        self.dataloader = DataLoader(self.dataset, batch_size=256, shuffle=False, num_workers=4)
        self.device = device
        # calculate accuracy
        self.accuracy = []
        self.model_name = model_chk_path.split("_")[2]
        
        #list index = epoch id
        # self.correct_per_epoch = []  # list of dicts: [{class_label: [indices]}, ...]
        # self.incorrect_per_epoch = []  # list of dicts: [{class_label: [indices]}, ...]

        for id,model in enumerate(self.epochs):
            correct = 0
            total = 0
            
            if (id == len(self.epochs)-1): #final epoch - track correct/incorrect ids
                self.correct_ids = defaultdict(list)
                self.incorrect_ids = defaultdict(list)

                with torch.no_grad():
                    for batch_idx, (images, labels) in enumerate(self.dataloader):
                        print("Batch idx:", batch_idx, "Batch size:", images.size(0), "labels:", set(labels.tolist()))
                        images = images.to(self.device)
                        labels = labels.to(self.device)
                        outputs = model(images)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()

                        
                        # Track correct/incorrect per class
                        for i, (pred, label) in enumerate(zip(predicted, labels)):
                            global_idx = batch_idx * self.dataloader.batch_size + i
                            label_val = label.item()
                            if pred == label:
                                self.correct_ids[label_val].append(global_idx)
                            else:
                                self.incorrect_ids[label_val].append(global_idx)
                    print(f"Total correct: {sum(len(v) for v in self.correct_ids.values())}, Total incorrect: {sum(len(v) for v in self.incorrect_ids.values())}")
                    print(total, correct)
                    
                    
            else:
                with torch.no_grad():
                    for images, labels in self.dataloader:
                        images = images.to(self.device)
                        labels = labels.to(self.device)
                        outputs = model(images)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()

            self.accuracy.append(100 * correct / total)

    def get_model_layer_names(self, epoch_idx):
        """Get names of convolutional, fully connected, and ReLU layers for CKA analysis"""
        return [l[0] for l in list(self.epochs[epoch_idx].named_modules()) if l[0] and (
                "conv" in l[0] or "fc" in l[0] or "relu" in l[0])]
    
    def get_model_layer_names_resnet(self, epoch_idx):
        """
        Return only high-level ResNet blocks to avoid redundancy.
        """
        allowed = {"conv1", "layer1", "layer2", "layer3", "layer4", "fc"}
        model = self.epochs[epoch_idx]

        return [name for name, _ in model.named_modules() if name in allowed]

    @staticmethod
    def _norm(x):
        return (x - x.min()) / (x.max() - x.min())

    def visualize_filters(self, show=True, save=False):
        models = [self.epochs[0], self.epochs[-1]]
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
            plt.savefig(os.path.join("figures", "filters", f"{self.model_name}_filters.jpg"))
        if show:
            plt.show()

def cka_comparison(epoch_idx1: int, ma1: ModelAnalysis, ma_layers1: list[str], epoch_idx2, ma2: ModelAnalysis = None,
                   ma_layers2: list[str] = None, plot=True, show=False, save=False):
    if ma2 is None:
        ma2 = ma1
    if ma_layers2 is None:
        ma_layers2 = ma_layers1
    with torch.amp.autocast('cuda', enabled=False):
        cka = CKA(
            ma1.epochs[epoch_idx1],
            ma2.epochs[epoch_idx2],
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
        col = plt.imshow(results['CKA'], vmin=0, vmax=1, cmap="coolwarm", origin="lower", aspect="auto")
        plt.colorbar(col)
        # annotate the heatmap
        for i in range(results['CKA'].shape[0]):
            for j in range(results['CKA'].shape[1]):
                plt.text(j, i, f"{results['CKA'][i, j]:.2f}", ha="center", va="center", color="black")

        plt.xticks(range(len(results['model2_layers'])), results['model2_layers'], rotation=90)
        plt.yticks(range(len(results['model1_layers'])), results['model1_layers'])
        plt.title(f"CKA comparison between {ma1.model_name} and {ma2.model_name}", fontsize=18, fontweight="bold")
        plt.xlabel(ma2.model_name, fontsize=15, fontweight="bold")
        plt.ylabel(ma1.model_name, fontsize=15, fontweight="bold")
        plt.tight_layout()
        os.makedirs(os.path.join("figures", "cka_simplecnn10"), exist_ok=True)
        if save:
            plt.savefig(
                os.path.join("figures", "cka_simplecnn10",
                             f"{ma1.model_name}_{ma2.model_name}_epoch1_{ma1.epochs_names[epoch_idx1]}_epoch2_{ma2.epochs_names[epoch_idx2]}.pdf")
            )
        if show:
            plt.show()
        else:
            plt.close('all')
    return results


# %%
device = get_device()
model_analysis_obj = []
for model_name in os.listdir(MODELS_FOLDER_PATH):
    print(model_name)
    model_analysis_obj.append(ModelAnalysis(model_name, device))

# %%
# Compare each model's last epoch (after training) with itself
# This measures the stability of the model representation on the test dataset
def cka_mean(results):
    """
    Given CKA results dict, return:
    - mean off-diagonal similarity
    """
    K = results['CKA']
    L = K.shape[0]

    # Off-diagonal similarity (excluding diagonal)
    off_diag_vals = [K[i, j] for i in range(L) for j in range(L) if i != j]
    mean_off_diag = np.mean(off_diag_vals)

    return mean_off_diag

cka_summary = []

for ma in tqdm(model_analysis_obj):
    epoch_idx = len(ma.epochs) - 1

    results = cka_comparison(
        epoch_idx, ma, ma.get_model_layer_names(epoch_idx),
        epoch_idx, ma, ma.get_model_layer_names(epoch_idx),
        plot=True, show=False, save=True
    )

    mean_off = cka_mean(results)

    cka_summary.append({
        "model": ma.model_name,
        "mean_off_diag": mean_off
    })

cka_df = pd.DataFrame(cka_summary)
cka_df.to_csv(OUTPUT_CSV, index=False)

