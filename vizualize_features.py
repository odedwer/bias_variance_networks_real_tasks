# In this file, we will visualize the learned features of the model, by plotting the activations of the first layer of the model for a given input image.
#
# We will use the `torchsummary` package to get the model summary and the `torchvision` package to load the image and preprocess it.
#
# We load a specific trained model from the `models` directory and use load an image from the CIFAR10 dataset in the `data` directory .
#
# Let's start by importing the necessary packages:
import torch
import matplotlib.pyplot as plt
from torchvision import transforms, datasets
from torchsummary import summary
from modules import AlexNet
from PIL import Image
import numpy as np
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# We define the path to the model:

model_path = r'/home/tomerach/Documents/bias_variance_networks_real_tasks/models/15-05-2025_16-52-51_SimpleCNN, BN=True, Bias=0.1/epoch-final.pth'

# We load the model:
checkpoint = torch.load(model_path, weights_only=True)

model = AlexNet(None, None)
model.load_state_dict(checkpoint)
# model to cpu
model.eval()

# We load images to train and valid loaders:
normalize = transforms.Normalize(
    mean=[0.4914, 0.4822, 0.4465],
    std=[0.2023, 0.1994, 0.2010],
)

# define transforms
train_transform = transforms.Compose([
    transforms.Resize((227, 227)),
    transforms.ToTensor(),
    normalize,
])

train_transform_no_norm = transforms.Compose([
    transforms.Resize((227, 227)),
    transforms.ToTensor(),
])
train_dataset_no_norm = datasets.CIFAR10(
    root='data', train=True,
    download=True, transform=train_transform_no_norm,
)

train_dataset = datasets.CIFAR10(
    root='data', train=True,
    download=True, transform=train_transform,
)

# pass the first image in the dataset to the model and get the activations of the first layer:
image, label = train_dataset[0]
image = image.unsqueeze(0)
activations = model.layer1(image)
activations = activations.squeeze(0).detach().numpy()

# We plot the activations near the input image:
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(train_dataset_no_norm[0][0].permute(1, 2, 0))
plt.title('Input Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(activations[0], cmap='jet')
plt.title('Activations of the first layer')
plt.axis('off')
plt.show()

#%% plot activations of all filters in the first layer
plt.figure(figsize=(20, 10))
for i in range(activations.shape[0]):
    plt.subplot(8, 12, i + 1)
    plt.imshow(activations[i], cmap='jet')
    plt.axis('off')
plt.show()
