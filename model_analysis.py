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

MODELS_FOLDER_PATH = "models/models_for_analysis_seed83/resnet10"
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

    def compute_saliency_maps(self, img_idx):
        """Compute saliency maps for the provided image index using the last epoch only.
        Returns a tuple (saliency_maps, predicted_idxs, epoch_names, image, true_label)
        """
        image, label = self.dataset[img_idx]
        image = image.to(self.device)
        true_label = self.dataset.number_label_map[label]

        idx = len(self.epochs) - 1
        model = self.epochs[idx]
        model.zero_grad()
        img_clone = image.clone().detach().to(self.device)
        img_clone.requires_grad = True

        output = model(img_clone[None, ...])
        # backward on predicted logit to get saliency
        output[0, output.argmax()].backward()
        saliency = self._norm(img_clone.grad.abs()[0])

        saliency_maps = [saliency]
        predicted_idxs = [output.argmax().item()]
        epoch_names = [self.epochs_names[idx]]

        return saliency_maps, predicted_idxs, epoch_names, image, true_label

    def visualize_saliency_map(self, img_idx, show=True, save=False, last=True):
        """
        Visualize saliency maps. By default (last=True) only visualizes the last epoch's map.
        To visualize all epochs set last=False or pass epoch_indices via compute_saliency_maps.
        """
        saliency_maps, predicted_idxs, epoch_names, image, true_label = self.compute_saliency_maps(img_idx, last=last)

        n = len(saliency_maps)
        n_rows, n_cols = np.sqrt(n).astype(int), np.ceil(n / np.sqrt(n)).astype(int)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 10))
        axes = axes.flatten()

        for i, saliency_map in enumerate(saliency_maps):
            axes[i].imshow(image[0].cpu().detach().numpy(), cmap="gray")
            c = axes[i].imshow(saliency_map.cpu().detach().numpy(), alpha=0.75, cmap="hot", vmin=0, vmax=1,
                               interpolation='bicubic')
            axes[i].set_title("Epoch: " + str(epoch_names[i]) + f", Acc:{self.accuracy[self.epochs_names.index(epoch_names[i])] if epoch_names[i] in self.epochs_names else 0:.2f}" + ", : " + (
                self.dataset.number_label_map[predicted_idxs[i]]))
            fig.colorbar(c, ax=axes[i], orientation="horizontal")

        for i in range(0, len(axes)):
            axes[i].axis("off")
        fig.suptitle(f"{self.model_name} Saliency map, img {img_idx}, class {true_label}")
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

    @staticmethod
    def saliency_entropy(saliency_map, eps=1e-12):
        """Compute entropy of a saliency map treating it as a probability distribution.
        Expects a NumPy array or torch tensor of non-negative values.
        Returns a scalar entropy (nats).
        """
        # Convert to numpy
        if isinstance(saliency_map, torch.Tensor):
            arr = saliency_map.detach().cpu().numpy()
        else:
            arr = np.array(saliency_map)
        flat = arr.ravel().astype(np.float64)
        # Ensure non-negative
        flat = np.clip(flat, 0, None)
        s = flat.sum()
        if s <= 0:
            return 0.0
        p = flat / (s + eps)
        return float(-np.sum(p * np.log(p + eps)))

    def compute_dataset_saliency_entropies(self, show_progress=True, saliency_maps):
        """Compute saliency entropies for every image in the test dataset using the last epoch.
        For images where multiple saliency maps are returned (future-proofing), the per-image
        entropy is the mean entropy across those maps.
        Returns a list of floats (one entropy per image) in dataset order.
        """
        entropies = []
        iterator = range(len(self.dataset))
        if show_progress:
            iterator = tqdm(iterator, desc=f"Entropies {self.model_name}")

        for img_idx in iterator:
            img_entropies = [self.saliency_entropy(sm) for sm in saliency_maps]
            if len(img_entropies) == 0:
                entropies.append(0.0)
            else:
                entropies.append(float(np.mean(img_entropies)))

        return entropies


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


def compute_models_entropy_stats(model_analysis_objs, show_progress=True):
    """For a list of ModelAnalysis instances, compute per-model average saliency entropy
    (averaged across images), and return a tuple (per_model_entropies, per_model_means,
    overall_mean, overall_variance).
    Always uses the last epoch and shows progress by default.
    """
    saliency_maps = self.compute_saliency_maps(img_idx)
    per_model_means = []
    per_model_entropies = {}
    iterator = (tqdm(model_analysis_objs, desc="Models") if show_progress else model_analysis_objs)
    for ma in iterator:
        entropies = ma.compute_dataset_saliency_entropies(show_progress=show_progress, saliency_maps)
        mean_entropy = float(np.mean(entropies)) if len(entropies) > 0 else 0.0
        per_model_means.append(mean_entropy)
        per_model_entropies[ma.model_name] = {
            "mean": mean_entropy,
            "n_images": len(entropies)
        }

    overall_mean = float(np.mean(per_model_means)) if len(per_model_means) > 0 else 0.0
    overall_variance = float(np.var(per_model_means)) if len(per_model_means) > 0 else 0.0
    return per_model_entropies, per_model_means, overall_mean, overall_variance


# %%
device = get_device()
model_analysis_obj = []
for model_name in os.listdir(MODELS_FOLDER_PATH):
    print(model_name)
    model_analysis_obj.append(ModelAnalysis(model_name, device))

# %%

# %%

# for model in model_analysis_obj:
#     model.visualize_filters(show=False, save=True)
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
    model = ma.epochs[epoch_idx] #last epoch by default


    result = {}
    correct_dict = ma.correct_ids
    incorrect_dict = ma.incorrect_ids
    # print("Correct dict:", correct_dict)
    # print("Incorrect dict:", incorrect_dict)

    all_classes = sorted(set(list(correct_dict.keys()) + list(incorrect_dict.keys()))) #gt labels
    #print("All classes:", all_classes)
    for cls in all_classes:
        print(f"Sampling for class {cls}:")
        corr_list = correct_dict.get(cls, [])
        incorr_list = incorrect_dict.get(cls, [])
        print(f"  Correct samples available: {len(corr_list)}, Incorrect samples available: {len(incorr_list)}")
        sampled_corr = random.sample(corr_list, min(n_per_class, len(corr_list))) if corr_list else []
        sampled_incorr = random.sample(incorr_list, min(n_per_class, len(incorr_list))) if incorr_list else []
        result[cls] = {"correct": sampled_corr, "incorrect": sampled_incorr}

    flattened = []
    for cls in sorted(result.keys()):
        flattened.extend(result[cls]["correct"])
        flattened.extend(result[cls]["incorrect"])

    return flattened


# for model in model_analysis_obj:
#     sampled_indices = sample_indices_per_class(model_analysis_obj[0].dataset, n_per_class=7, ma=model, epoch_idx=-1)
#     for idx in sampled_indices:
#         model.visualize_saliency_map(idx, show=False, save=True)
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

