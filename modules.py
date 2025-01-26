import torch
import torch.nn as nn


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
                nn.init.kaiming_normal_(m.weight)
                nn.init.normal_(m.bias, 0, self.b_scale)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                nn.init.normal_(m.bias, 0, self.b_scale)
            elif isinstance(m, nn.BatchNorm2d):
                # nn.init.kaiming_normal_(m.weight)
                nn.init.normal_(m.bias, 0, self.b_scale)


# Write alexnet with reinitialization

class AlexNet(BiasVarianceNetwork):
    def __init__(self, w_scale, b_scale, num_classes=100, freeze_bias: [bool, list] = False, **kwargs):
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

        if freeze_bias:
            self.layer1[0].bias.requires_grad = False
            self.layer2[0].bias.requires_grad = False
            self.layer3[0].bias.requires_grad = False
            self.layer4[0].bias.requires_grad = False
            self.layer5[0].bias.requires_grad = False
            self.fc[1].bias.requires_grad = False
            self.fc1[1].bias.requires_grad = False
            self.fc2[0].bias.requires_grad = False

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
