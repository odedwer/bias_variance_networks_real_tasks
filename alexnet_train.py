import datetime
import os
from itertools import product

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
from torchvision import transforms
from tqdm import tqdm
import pandas as pd

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from modules import AlexNet


def get_train_valid_loader(data_dir,
                           batch_size,
                           augment,
                           random_seed,
                           valid_size=0.1,
                           shuffle=True):
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )

    # define transforms
    valid_transform = transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
        normalize,
    ])
    if augment:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((227, 227)),
            transforms.ToTensor(),
            normalize,
        ])

    # load the dataset
    train_dataset = datasets.CIFAR10(
        root=data_dir, train=True,
        download=True, transform=train_transform,
    )

    valid_dataset = datasets.CIFAR10(
        root=data_dir, train=True,
        download=True, transform=valid_transform,
    )

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler)

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, sampler=valid_sampler)

    return (train_loader, valid_loader)


def get_test_loader(data_dir,
                    batch_size,
                    shuffle=True):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    # define transform
    transform = transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
        normalize,
    ])

    dataset = datasets.CIFAR10(
        root=data_dir, train=False,
        download=True, transform=transform,
    )

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle
    )

    return data_loader


def get_summary_writer(model_name, **kwargs):
    timestamp = str(datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S"))
    exp_name = os.path.join(timestamp, model_name,
                            *(f"{k}_{f'{v:.2g}' if isinstance(v, str) else str(v)}" for k, v in kwargs.items()))
    return SummaryWriter(log_dir=os.path.join("runs", exp_name)), exp_name


# CIFAR10 dataset
train_loader, valid_loader = get_train_valid_loader(data_dir='./data', batch_size=64,
                                                    augment=False, random_seed=1)

test_loader = get_test_loader(data_dir='./data',
                              batch_size=64)


def get_params(freeze_bias, num_classes, num_epochs, batch_sizes, learning_rate, b_scales, w_scales):
    params = []
    for fb in freeze_bias:
        for nc in num_classes:
            for ne in num_epochs:
                for bs in batch_sizes:
                    for lr in learning_rate:
                        params.append([False, fb, nc, ne, bs, lr, 1, np.sqrt(5)])
                        for b_scale in b_scales:
                            for w_scale in w_scales:
                                params.append([True, fb, nc, ne, bs, lr, b_scale, w_scale])
    return params


freeze_bias = [False, True]
num_classes = [10]
num_epochs = [30]
batch_sizes = [64, 128]
learning_rate = [1e-4, 5e-4, 1e-3, 5e-3]
b_scales = [0.1, 1, 5, 10]
w_scales = [np.sqrt(5)]

params = get_params(freeze_bias, num_classes, num_epochs, batch_sizes, learning_rate, b_scales, w_scales)

params_df = pd.DataFrame(params,
                         columns=["reinitialize", "freeze_bias", "num_classes", "num_epochs", "batch_size", "lr",
                                  "b_scale", "w_scale"])

for i, param in params_df.iterrows():
    torch.manual_seed(42)
    model = AlexNet(**param.to_dict()).to(device)
    if param.reinitialize:
        model.reinitialize(seed=1)

    torch.manual_seed(42)
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=param.lr, momentum=0.9)

    writer, exp_name = get_summary_writer("alexnet", **param.to_dict())
    # Train the model
    total_step = len(train_loader)

    for epoch in tqdm(range(param.num_epochs)):
        for i, (images, labels) in enumerate(train_loader):
            # Move tensors to the configured device
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            writer.add_scalar("Loss/train", loss, epoch)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
        #       .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

        # Validation
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in valid_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                # add validation loss to tensorboard
                loss = criterion(outputs, labels)
                writer.add_scalar("Loss/validation", loss, epoch)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                del images, labels, outputs
            writer.add_scalar("Accuracy/validation", 100 * correct / total, epoch)

    writer.flush()
    writer.close()

    os.makedirs("models", exist_ok=True)
    # find the name for the model
    torch.save(model.state_dict(), os.path.join("models", exp_name) + ".pth")
