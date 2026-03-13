"""
ResNet18 implementation from scratch for facial expression recognition.

Implements:
- BasicBlock: Residual block with optional batch normalization
- ResNet: ResNet18 architecture with customizable batch norm and bias

References:
- Deep Residual Learning for Image Recognition (https://arxiv.org/pdf/1512.03385v1.pdf)
"""

import torch.nn as nn
import torch

from torch import Tensor
from typing import Type


class BasicBlock(nn.Module):
    """
    Residual block for ResNet18.
    
    Performs two convolutions with optional batch normalization and a skip connection.
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        stride (int): Stride for first convolution (default: 1)
        expansion (int): Channel expansion factor (default: 1 for ResNet18)
        downsample (nn.Module, optional): Module to adjust skip connection dimensions
        bias (bool): Whether to use bias in convolutions (default: False)
        bn (bool): Whether to use batch normalization (default: True)
    """
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            stride: int = 1,
            expansion: int = 1,
            downsample: nn.Module = None,
            bias: bool = False,
            bn: bool = True
    ) -> None:
        super(BasicBlock, self).__init__()
        # Multiplicative factor for the subsequent conv2d layer's output channels.
        # It is 1 for ResNet18 and ResNet34.
        self.bias = bias
        self.bn = bn
        self.expansion = expansion
        self.downsample = downsample
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=bias,
        )
        self.bn1 = nn.BatchNorm2d(out_channels) if self.bn else nn.Identity()
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels * self.expansion,
            kernel_size=3,
            padding=1,
            bias=bias,
        )
        self.bn2 = nn.BatchNorm2d(out_channels * self.expansion) if self.bn else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    """
    ResNet18 architecture for image classification.
    
    Implements ResNet18 with support for disabling batch normalization and 
    customizing bias initialization for variance study.
    
    Args:
        img_channels (int): Number of input image channels (default: 1 for grayscale)
        num_layers (int): Number of layers (currently only 18 supported)
        block (Type): Residual block class (default: BasicBlock)
        num_classes (int): Number of output classes (default: 2)
        bias (bool): Whether to use bias in convolutions (default: False)
        bn (bool): Whether to use batch normalization (default: True)
    """
    def __init__(
            self,
            img_channels: int=1,
            num_layers: int = 18,
            block: Type[BasicBlock] = BasicBlock,
            num_classes: int = 2,
            bias: bool = False,
            bn: bool = True
    ) -> None:
        super(ResNet, self).__init__()
        self.bias = bias
        self.bn = bn
        if num_layers == 18:
            # The following `layers` list defines the number of `BasicBlock`
            # to use to build the network and how many basic blocks to stack
            # together.
            layers = [2, 2, 2, 2]
            self.expansion = 1

        self.in_channels = 64
        # All ResNets (18 to 152) contain a Conv2d => BN => ReLU for the first
        # three layers. Here, kernel size is 3.
        self.conv1 = nn.Conv2d(
            in_channels=img_channels,
            out_channels=self.in_channels,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=bias,
        )
        self.bn1 = nn.BatchNorm2d(self.in_channels) if self.bn else nn.Identity()
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * self.expansion, num_classes)

    def _make_layer(
            self, block: Type[BasicBlock], out_channels: int, blocks: int, stride: int = 1
    ) -> nn.Sequential:
        downsample = None
        if stride != 1:
            """
            This should pass from `layer2` to `layer4` or
            when building ResNets50 and above. Section 3.3 of the paper
            Deep Residual Learning for Image Recognition
            (https://arxiv.org/pdf/1512.03385v1.pdf).
            """
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    out_channels * self.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=self.bias,
                ),
                nn.BatchNorm2d(out_channels * self.expansion) if self.bn else nn.Identity(),
            )
        layers = []
        layers.append(
            block(self.in_channels, out_channels, stride, self.expansion, downsample, self.bias, self.bn)
        )
        self.in_channels = out_channels * self.expansion

        for i in range(1, blocks):
            layers.append(
                block(self.in_channels, out_channels, expansion=self.expansion, bias=self.bias, bn=self.bn)
            )
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # The spatial dimension of the final layer's feature
        # map should be (7, 7) for all ResNets.
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
