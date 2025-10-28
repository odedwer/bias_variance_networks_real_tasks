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
        self.dataset = FER2013Dataset('data/face-expression/test', transform=test_transforms)
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
    if ma2 is None:
        ma2 = ma1
    if ma_layers2 is None:
        ma_layers2 = ma_layers1
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

if __name__ == '__main__':

    # %%
    device = get_device()
    model_analysis_obj = []
    for model_name in os.listdir("models"):
        print(model_name)
        model_analysis_obj.append(ModelAnalysis(model_name, device))
    # %%
    dataset = FER2013Dataset('data/face-expression/test', transform=test_transforms)
    for i in range(7):
        choice_idx = np.random.choice(np.where(dataset.labels == i)[0],4,False)
        for idx in choice_idx:
            for model in model_analysis_obj:
                model.visualize_saliency_map(idx, show=False, save=True)

    for model in model_analysis_obj:
        model.visualize_filters(show=False, save=True)

    # %%
    model_pairs = combinations_with_replacement(range(len(model_analysis_obj)), 2)
    for ma1_idx, ma2_idx in tqdm(list(model_pairs)):
        ma1 = model_analysis_obj[ma1_idx]
        ma2 = model_analysis_obj[ma2_idx]
        for epoch_idx1 in range(len(ma1.models)):
            for epoch_idx2 in range(len(ma2.models)):
                results = cka_comparison(epoch_idx1, ma1, ma1.get_model_layer_names(epoch_idx1), epoch_idx2,
                                         ma2, ma2.get_model_layer_names(epoch_idx2),
                                         show=False, save=True)
