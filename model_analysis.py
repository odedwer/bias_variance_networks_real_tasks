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

from ResNet import ResNet
from face_recognition_model_comparison import SimpleCNN, test_transforms, FER2013Dataset

MODELS_FOLDER_PATH = "models/models_for_analysis_seed83"
classes=["fear","angry"]

class ModelAnalysis:
    """
    a model after training different epochs
    """
    def __init__(self, model_chk_path, device):
        self.epochs_names = []
        model_chkpoints = os.listdir(os.path.join(MODELS_FOLDER_PATH, model_chk_path))
        self.params = pd.read_csv(os.path.join("runs", model_chk_path, "params.csv"))
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
        for model in self.epochs:
            correct = 0
            total = 0
            #TODO continue here
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

    def visualize_saliency_map(self, img_idx, show=True, save=False):
        image, label = self.dataset[img_idx]
        image = image.to(self.device)
        image.requires_grad = True
        label = self.dataset.number_label_map[label] #the true label
        saliency_maps = []
        for model in self.epochs:
            model.zero_grad()
            image.requires_grad = True
            output = model(image[None, ...].to(self.device))
            output[0, output.argmax()].backward()
            saliency_maps.append(self._norm(image.grad.abs()[0]))
            # plot the saliency map on top of the image
        n_rows, n_cols = np.sqrt(len(saliency_maps)).astype(int), np.ceil(
            len(saliency_maps) / np.sqrt(len(saliency_maps))).astype(int)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 10))
        axes = axes.flatten()
        # share all heatmap colorbars
        for i, saliency_map in enumerate(saliency_maps):
            axes[i].imshow(image[0].cpu().detach().numpy(), cmap="gray")
            # calculate smooth
            c = axes[i].imshow(saliency_map.cpu().detach().numpy(), alpha=0.75, cmap="hot", vmin=0, vmax=1,
                               interpolation='bicubic')

            axes[i].set_title("Epoch: " + str(self.epochs_names[i]) + f", Acc:{self.accuracy[i]:.2f}" + ", : " + (
                self.dataset.number_label_map[output.argmax().item()])) #the predicted label
            fig.colorbar(c, ax=axes[i], orientation="horizontal")
            # remove axis

        for i in range(0, len(axes)):
            axes[i].axis("off")
        fig.suptitle(f"{self.model_name} Saliency map, img {img_idx}, class {label}")
        fig.tight_layout()
        if save:
            os.makedirs(os.path.join("figures", "saliency_maps", self.model_name), exist_ok=True)
            plt.savefig(
                os.path.join("figures", "saliency_maps", self.model_name, f"{self.model_name}_img_{img_idx}.pdf")
            )
        if show:
            plt.show()

    def get_model_layer_names(self, epoch_idx):
        return [l[0] for l in list(self.epochs[epoch_idx].named_modules()) if l[0] and (
                "conv" in l[0] or "fc" in l[0] or "relu" in l[0])]


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
        os.makedirs(os.path.join("figures", "cka"), exist_ok=True)
        if save:
            plt.savefig(
                os.path.join("figures", "cka",
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

for model in model_analysis_obj:
    model.visualize_filters(show=False, save=True)
#  %%
import random
from collections import defaultdict

def sample_indices_per_class(dataset, n_per_class=3, seed=42, ma: ModelAnalysis = None, epoch_idx: int = -1):
    """
    If ma is None: same behavior as before -> returns a flat list of sampled indices (n_per_class per class).
    If ma is provided: returns a dict mapping class_index -> {"correct": [...], "incorrect": [...]} where each list
    contains up to n_per_class sampled indices for that class (according to model `ma.epochs[epoch_idx]` predictions).
    """
    random.seed(seed)

    if ma is None:
        class_to_indices = defaultdict(list)
        for idx, (_, label) in enumerate(dataset):
            class_to_indices[label].append(idx)
        sampled_indices = []
        for indices in class_to_indices.values():
            sampled_indices.extend(random.sample(indices, min(n_per_class, len(indices))))
        return sampled_indices

    # Use the provided ModelAnalysis to separate correct / incorrect samples per class
    model = ma.epochs[epoch_idx]
    model.eval()
    device = ma.device

    correct = defaultdict(list)
    incorrect = defaultdict(list)

    with torch.no_grad():
        for idx in range(100):
            image, label = dataset[idx]
            # image is expected to be a tensor already transformed by dataset
            out = model(image.unsqueeze(0).to(device))
            pred = out.argmax(dim=1).item()
            if pred == label:
                correct[label].append(idx)
            else:
                incorrect[label].append(idx)

    result = {}
    all_classes = sorted(set(list(correct.keys()) + list(incorrect.keys())))
    for cls in all_classes:
        corr_list = correct.get(cls, [])
        incorr_list = incorrect.get(cls, [])
        sampled_corr = random.sample(corr_list, min(n_per_class, len(corr_list))) if corr_list else []
        sampled_incorr = random.sample(incorr_list, min(n_per_class, len(incorr_list))) if incorr_list else []
        result[cls] = {"correct": sampled_corr, "incorrect": sampled_incorr}

    return result


for model in model_analysis_obj:
    sampled_indices = sample_indices_per_class(model_analysis_obj[0].dataset, n_per_class=7, ma=model, epoch_idx=-1)
    for idx in sampled_indices:
        model.visualize_saliency_map(idx, show=False, save=True)
# %%
model_pairs = combinations_with_replacement(range(len(model_analysis_obj)), 2)
for ma1_idx, ma2_idx in tqdm(list(model_pairs)):
    ma1 = model_analysis_obj[ma1_idx]
    ma2 = model_analysis_obj[ma2_idx]
    for epoch_idx1 in range(len(ma1.epochs)):
        for epoch_idx2 in range(len(ma2.epochs)):
            results = cka_comparison(epoch_idx1, ma1, ma1.get_model_layer_names(epoch_idx1), epoch_idx2,
                                     ma2, ma2.get_model_layer_names(epoch_idx2),
                                     show=False, save=True)
