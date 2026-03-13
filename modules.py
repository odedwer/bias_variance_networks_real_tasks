import os

import numpy as np
import torch
import torch.nn as nn
from torch.nn import Sequential
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.io import decode_image
import pandas as pd
from utils import calculate_conv_width_height


class DeviceDataLoader:
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def _to_device(self, x, y):
        return x.to(self.device), y.to(self.device)

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        for b in self.dl:
            yield (self._to_device(*b))


class FER2013(Dataset):
    def __init__(self, img_dir):
        self.img_dir = img_dir
        self.unique_labels = os.listdir(img_dir)
        self._img_count = []
        self.label_number_map = {}
        self.number_label_map = {}
        for i, label in enumerate(self.unique_labels):
            self._img_count.append(len(os.listdir(os.path.join(img_dir, label))))
            self.number_label_map[i] = label
            self.label_number_map[label] = i
        self.labels = []
        for l, c in zip(self.unique_labels, self._img_count):
            self.labels.extend([self.label_number_map[l]] * c)
            # self.labels.extend([l] * c)

        self.images = []
        for label in self.unique_labels:
            self.images.extend([os.path.join(self.img_dir, label, img) for img in
                                os.listdir(os.path.join(img_dir, label))])
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])
        self.labels = np.array(self.labels)

    def __len__(self):
        return sum(self._img_count)

    def __getitem__(self, idx):
        return self.transform(
            decode_image(self.images[idx]).numpy().astype(np.float32).reshape((48, 48, 1)) / 255).type(torch.float32), \
            self.labels[idx]

    def get_class_weights(self):
        return 1 / np.array(self._img_count)


# Base network class with reinitialization method for setting bias variance
class BiasVarianceNetwork(nn.Module):
    def __init__(self, w_scale, b_scale, **kwargs):
        super(BiasVarianceNetwork, self).__init__()
        self._layers = None
        self.w_scale = w_scale
        self.b_scale = b_scale
        self._handles = []
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.freeze = None

    def set_activations_hook(self, activations):
        def hook_generator(name, activations):
            def hook(model, input, output):
                activations[name] = output.detach().numpy()

            return hook

        self._handles = []
        for name, m in self._layers.named_modules():
            self._handles.append(m.register_forward_hook(hook_generator(name, activations)))

    def remove_activations_hook(self):
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def get_out_activation(self):
        return self._layers[-1]

    def forward(self, x):
        return self._layers(x)

    def partial_reinitialize(self, layers: list, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        for n, m in self.named_modules():
            try:
                name = n.split('.')
            except Exception:
                name = n
            if name in layers:
                m.bias.requires_grad = False
                if isinstance(m, nn.Conv2d):
                    self._reinitialize_conv(m)
                elif isinstance(m, nn.Linear):
                    self._reinitialize_linear(m)
                elif isinstance(m, nn.BatchNorm2d):
                    self._reinitialize_batchnorm(m)

    def reinitialize(self, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                self._reinitialize_linear(m)
            elif isinstance(m, nn.Conv2d):
                self._reinitialize_conv(m)
            elif isinstance(m, nn.BatchNorm2d):
                self._reinitialize_batch_norm(m)

    def _reinitialize_batchnorm(self, m):
        # nn.init.kaiming_normal_(m.weight)
        nn.init.normal_(m.bias, 0, self.b_scale)

    def _reinitialize_conv(self, m):
        nn.init.kaiming_normal_(m.weight, self.w_scale)
        nn.init.normal_(m.bias, 0, self.b_scale)

    def _reinitialize_linear(self, m):
        nn.init.kaiming_normal_(m.weight, self.w_scale)
        nn.init.normal_(m.bias, 0, self.b_scale)

    def freeze_bias(self):
        if isinstance(self.freeze, bool) and self.freeze:
            for n, m in self.named_modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    m.bias.requires_grad = False
        elif isinstance(self.freeze, list):
            for n, m in self.named_modules():
                try:
                    name = n.split('.')
                except Exception:
                    name = n
                if name in self.freeze:
                    m.bias.requires_grad = False

class SimpleCNN(BiasVarianceNetwork):
    def __init__(self, name, w_scale, b_scale, n_blocks_increasing=3, n_block_decreasing=1,
                 conv_params=None, pool_params=None,should_batchnorm=True, **kwargs):
        super(SimpleCNN, self).__init__(w_scale, b_scale)
        if conv_params is None:
            conv_params = {"kernel_size": 3, "padding": 1, "stride": 1}
        if pool_params is None:
            pool_params = {"kernel_size": 2, "stride": 2}
        self.conv_params = conv_params
        self.pool_params = pool_params
        self.name = name
        self._block_count = 0
        self._layers = Sequential()
        self.batchnorm = should_batchnorm
        self._build_network(n_block_decreasing, n_blocks_increasing)

    def _build_network(self, n_block_decreasing, n_blocks_increasing):
        in_channels, out_channels = 1, 32
        for i in range(n_blocks_increasing):
            block = self._get_block(in_channels, out_channels)
            self._layers.add_module(f"block{self._block_count}", block)
            in_channels = out_channels
            out_channels *= 2
        out_channels //= 2
        in_channels = out_channels
        out_channels //= 2
        for i in range(n_block_decreasing):
            block = self._get_block(in_channels, out_channels)
            self._layers.add_module(f"block{self._block_count}", block)
            in_channels = out_channels
            out_channels //= 2
        fc = Sequential()
        fc.add_module("flatten", nn.Flatten())
        size = int(((48 / (2 ** (n_block_decreasing + n_blocks_increasing))) ** 2) * in_channels)
        fc.add_module("fc1", nn.Linear(size, 128))
        fc.add_module("fc1_relu", nn.ReLU())
        fc.add_module("fc2", nn.Linear(128, 64))
        fc.add_module("fc2_relu", nn.ReLU())
        fc.add_module("fc3", nn.Linear(64, 7))
        self._layers.add_module("fc", fc)
        self._layers.add_module("softmax", nn.Softmax(-1))

    def _get_block(self, in_channels, out_channels):
        self._block_count += 1
        block = Sequential()
        block.add_module(f"conv{self._block_count}", nn.Conv2d(in_channels, out_channels, **self.conv_params))
        if self.batchnorm:
            block.add_module(f"batchnorm{self._block_count}", nn.BatchNorm2d(out_channels))
        block.add_module(f"activation{self._block_count}", nn.Tanh())
        block.add_module(f"pool{self._block_count}", nn.MaxPool2d(**self.pool_params))
        block.add_module(f"dropout{self._block_count}", nn.Dropout(0.25))
        return block
