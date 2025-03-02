import torch
import torch.nn as nn
from torch.nn import Sequential


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


# Write alexnet with reinitialization

class AlexNet(BiasVarianceNetwork):
    def __init__(self, w_scale, b_scale, num_classes=10, freeze_bias: [bool, list] = False, **kwargs):
        super(AlexNet, self).__init__(w_scale, b_scale, num_classes=num_classes)
        self.layer1 = nn.Sequential()
        self.layer1.add_module("l1_conv", nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0))
        self.layer1.add_module("l1_batchnorm", nn.BatchNorm2d(96))
        self.layer1.add_module("l1_ReLU", nn.ReLU())
        self.layer1.add_module("l1_maxpool", nn.MaxPool2d(kernel_size=3, stride=2))

        self.layer2 = nn.Sequential()
        self.layer2.add_module("l2_conv", nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2))
        self.layer2.add_module("l2_batchnorm", nn.BatchNorm2d(256))
        self.layer2.add_module("l2_ReLU", nn.ReLU())
        self.layer2.add_module("l2_maxpool", nn.MaxPool2d(kernel_size=3, stride=2))

        self.layer3 = nn.Sequential()
        self.layer3.add_module("l3_conv", nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1))
        self.layer3.add_module("l3_batchnorm", nn.BatchNorm2d(384))
        self.layer3.add_module("l3_ReLU", nn.ReLU())

        self.layer4 = nn.Sequential()
        self.layer4.add_module("l4_conv", nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1))
        self.layer4.add_module("l4_batchnorm", nn.BatchNorm2d(384))
        self.layer4.add_module("l4_ReLU", nn.ReLU())

        self.layer5 = nn.Sequential()
        self.layer5.add_module("l5_conv", nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1))
        self.layer5.add_module("l5_batchnorm", nn.BatchNorm2d(256))
        self.layer5.add_module("l5_ReLU", nn.ReLU())
        self.layer5.add_module("l5_maxpool", nn.MaxPool2d(kernel_size=3, stride=2))

        self.fc = nn.Sequential()
        self.fc.add_module("fc_dropout", nn.Dropout(0.5))
        self.fc.add_module("fc", nn.Linear(9216, 4096))
        self.fc.add_module("fc_ReLU", nn.ReLU())

        self.fc1 = nn.Sequential()
        self.fc1.add_module("fc1_dropout", nn.Dropout(0.5))
        self.fc1.add_module("fc1", nn.Linear(4096, 4096))
        self.fc1.add_module("fc1_ReLU", nn.ReLU())

        self.fc2 = nn.Sequential()
        self.fc2.add_module("fc2", nn.Linear(4096, num_classes))
        self.fc2.add_module("softmax", nn.Softmax(-1))

        if freeze_bias:
            conv_blocks = [self.layer1, self.layer2, self.layer3, self.layer4, self.layer5]
            fc_layers = [self.fc, self.fc1, self.fc2]
            if isinstance(freeze_bias, bool):
                for block in conv_blocks:
                    block[0].bias.requires_grad = False
                    block[1].bias.requires_grad = False
                for layer in fc_layers:
                    layer[1].bias.requires_grad = False
                self.fc[1].bias.requires_grad = False
                self.fc1[1].bias.requires_grad = False
                self.fc2[0].bias.requires_grad = False
            else:
                assert len(freeze_bias) == 8

    def __getitem__(self, item):
        for name, m in self.named_modules():
            if item in name:
                return m

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

    def get_out_activation(self):
        return self.fc2


class SimpleCNN(BiasVarianceNetwork):
    def __init__(self, name, w_scale, b_scale, n_blocks_increasing=3, n_block_decreasing=1, **kwargs):
        super(SimpleCNN, self).__init__(w_scale, b_scale)
        self.name = name
        self._block_count = 0
        self._build_network(n_block_decreasing, n_blocks_increasing)

    def _build_network(self, n_block_decreasing, n_blocks_increasing):
        in_channels, out_channels = 3, 32
        for i in range(n_blocks_increasing):
            block = self._get_block(in_channels, out_channels)
            self._layers.add_module(f"block{self._block_count}", block)
            in_channels = out_channels
            out_channels *= 2
        in_channels = out_channels
        out_channels //= 2
        for i in range(n_block_decreasing):
            block = self._get_block(in_channels, out_channels)
            setattr(self, f"block{i}", block)
            in_channels = out_channels
            out_channels //= 2
        fc = Sequential()
        fc.add_module("flatten", nn.Flatten())
        fc.add_module("fc1", nn.Linear(in_channels, 128))
        fc.add_module("fc1_relu", nn.ReLU())
        fc.add_module("fc2", nn.Linear(128, 64))
        fc.add_module("fc2_relu", nn.ReLU())
        fc.add_module("fc3", nn.Linear(64, 7))
        self._layers.add_module("fc", fc)
        self._layers.add_module("softmax", nn.Softmax(-1))

    def _get_block(self, in_channels, out_channels):
        self._block_count += 1
        block = Sequential()
        block.add_module(f"conv{self._block_count}", nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        block.add_module(f"activation{self._block_count}", nn.Tanh())
        block.add_module(f"pool{self._block_count}", nn.MaxPool2d(2, 2))
        block.add_module(f"dropout{self._block_count}", nn.Dropout(0.25))
        return block
