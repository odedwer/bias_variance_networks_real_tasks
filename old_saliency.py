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

    def compute_dataset_saliency_entropies(self, saliency_maps):
        """Compute saliency entropies for every image in the test dataset using the last epoch.
        For images where multiple saliency maps are returned (future-proofing), the per-image
        entropy is the mean entropy across those maps.
        Returns a list of floats (one entropy per image) in dataset order.
        """
        entropies = []
        iterator = range(len(self.dataset))
        iterator = tqdm(iterator, desc=f"Entropies {self.model_name}")

        for img_idx in iterator:
            img_entropies = [self.saliency_entropy(sm) for sm in saliency_maps]
            if len(img_entropies) == 0:
                entropies.append(0.0)
            else:
                entropies.append(float(np.mean(img_entropies)))

        return entropies
def compute_models_entropy_stats(model_analysis_objs):
    """For a list of ModelAnalysis instances, compute per-model average saliency entropy
    (averaged across images), and return a tuple (per_model_entropies, per_model_means,
    overall_mean, overall_variance).
    Always uses the last epoch and shows progress by default.
    """
    saliency_maps = self.compute_saliency_maps(img_idx)
    per_model_means = []
    per_model_entropies = {}
    iterator = (tqdm(model_analysis_objs, desc="Models"))
    for ma in iterator:
        entropies = ma.compute_dataset_saliency_entropies(saliency_maps)
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