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
from utils import get_device

from ResNet import ResNet
from face_recognition_model_comparison import SimpleCNN, test_transforms, FER2013Dataset


class ModelAnalysis:
    def __init__(self, model_chk_path, device, max_epoch):
        self.epochs = []
        model_chkpoints = os.listdir(os.path.join("models", model_chk_path))
        if "SimpleCNN" in model_chk_path:
            self.models = [SimpleCNN() for _ in range(len(model_chkpoints))]
        elif "ResNet" in model_chk_path:
            self.models = [ResNet(bn="BN=True" in model_chk_path, bias="Bias=None" not in model_chk_path) for _ in
                           range(len(model_chkpoints))]
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
        self.dataloader = DataLoader(self.dataset, batch_size=256, shuffle=False, num_workers=4)
        self.device = device
        # calculate accuracy
        self.accuracy = []
        self.model_name = model_chk_path.split("_")[2]
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

    def visualize_filters(self):
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
        plt.show()

    def visualize_saliency_map(self, img_idx):
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
        axes = axes.flatten()
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
        plt.show()

    def get_model_layer_names(self, epoch_idx):
        return [l[0] for l in list(self.models[epoch_idx].named_modules()) if l[0] and (
                "conv" in l[0] or "fc" in l[0] or "relu" in l[0])]


def cka_comparison(epoch_idx: int, ma1: ModelAnalysis, ma_layers1: list[str], ma2: ModelAnalysis = None,
                   ma_layers2: list[str] = None,plot=True):
    if ma2 is None:
        ma2 = ma1
    if ma_layers2 is None:
        ma_layers2 = ma_layers1
    with torch.cuda.amp.autocast(enabled=False):
        cka = CKA(
            ma1.models[epoch_idx],
            ma2.models[epoch_idx],
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
        fig = plt.figure(figsize=(15,15))
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
        plt.show()
    return results


# %%
a = ModelAnalysis("05-03-2025_12-49-22_ResNet, BN=True, Bias=None", get_device(), 60)

b = ModelAnalysis("05-03-2025_12-23-08_ResNet, BN=False, Bias=None", get_device(), 60)

c = ModelAnalysis("05-03-2025_12-29-19_ResNet, BN=False, Bias=10.0", get_device(), 60)
# %%
a.visualize_saliency_map(400)
b.visualize_saliency_map(400)
c.visualize_saliency_map(400)
# %%
epoch_idx=-1
results = cka_comparison(epoch_idx, b, b.get_model_layer_names(epoch_idx), c,c.get_model_layer_names(epoch_idx))
# %%

