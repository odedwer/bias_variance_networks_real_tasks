import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
from torchvision import transforms
from tqdm import tqdm

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


# CIFAR10 dataset
train_loader, valid_loader = get_train_valid_loader(data_dir='./data', batch_size=64,
                                                    augment=False, random_seed=1)

test_loader = get_test_loader(data_dir='./data',
                              batch_size=64)

num_classes = 10
num_epochs = 20
batch_size = 64
learning_rate = 0.005

for reinitialize, b_scale in [(False, 1), (True, 0.1), (True, 1), (True, 10)]:
    w_scale = np.sqrt(5)
    torch.manual_seed(42)
    model = AlexNet(w_scale, b_scale, num_classes).to(device)
    if reinitialize:
        model.reinitialize(seed=1)


    torch.manual_seed(42)
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    params = {
            'lr': learning_rate, 'batch size': batch_size,
            'b_scale': b_scale, 'w_scale': w_scale, 'loss': criterion.__class__.__name__,
            'optimizer': optimizer.__class__.__name__, 'reinitialize': reinitialize
        }
    name = "alexnet_" + "_".join([f"{k}={v}" for k, v in params.items()])
    print("Model params:\n", name)
    writer = SummaryWriter("runs/" + name)
    # writer.add_hparams(
    #     {
    #         'lr': learning_rate, 'batch size': batch_size,
    #         'b_scale': b_scale, 'w_scale': w_scale, 'loss': criterion.__class__.__name__,
    #         'optimizer': optimizer.__class__.__name__, 'reinitialize': reinitialize
    #     },
    #     {}
    # )

    # Train the model
    total_step = len(train_loader)

    for epoch in range(num_epochs):
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

        print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
              .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

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
            print('Accuracy of the network on the {} validation images: {} %'.format(5000, 100 * correct / total))

    writer.flush()
    writer.close()

    os.makedirs("models", exist_ok=True)
    # find the name for the model
    model_name = "alexnet"
    # if exists, increment the model name
    i = 1
    while os.path.exists(f"models/{model_name}" + ".pth"):
        model_name = "alexnet" + str(i)
        i += 1

    torch.save(model.state_dict(), "models/" + model_name + ".pth")
