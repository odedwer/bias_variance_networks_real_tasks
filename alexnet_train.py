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
from modules import AlexNet, DeviceDataLoader
from clusterify import clusterify, finalize

def get_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device: " + str(device))
    return device

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
                            *(f"{k}_{f'{v:.2g}' if isinstance(v, float) else str(v)}" for k, v in
                              kwargs.items())).replace("\\", "/")
    return SummaryWriter(log_dir=os.path.join("runs", exp_name).replace("\\", "/")), exp_name




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


# %% constants
FREEZE_BIAS_LIST = [False, True]
NUM_CLASSES_LIST = [10]
NUM_EPOCHS_LIST = [20]
BATCH_SIZE_LIST = [64, 128]
LR_LIST = [1e-4, 5e-3, 1e-3]
BIAS_VAR_LIST = [0.1, 1, 5, 10]
WEIGHT_VAR_LIST = [0.1, np.sqrt(5), 1]


def main():
    params = get_params(FREEZE_BIAS_LIST, NUM_CLASSES_LIST, NUM_EPOCHS_LIST, BATCH_SIZE_LIST, LR_LIST, BIAS_VAR_LIST,
                        WEIGHT_VAR_LIST)

    params_df = pd.DataFrame(params,
                             columns=["reinitialize", "freeze_bias", "num_classes", "num_epochs", "batch_size", "lr",
                                      "b_scale", "w_scale"])
    os.makedirs("models", exist_ok=True)

    for i, param in params_df.iterrows():
        end_to_end_model_train(i, param)

@clusterify(chunk_size=1,n_jobs=20,
            job_script_prologue = ['module load cuda', 'module load py-torch'],
            memory='16GB',walltime='1:00:00',
            job_extra_directives=['--gres=gpu:a30:1', '--job-name=alexnet', '--output=~/lab/logs/alexnet-%j.out'])
def end_to_end_model_train(i, param):
    device = get_device()
    criterion, model, optimizer, test_loader, train_loader, valid_loader = init_training(device, param)
    writer, exp_name = get_summary_writer("alexnet", **param.to_dict())
    os.makedirs(os.path.join("models", exp_name), exist_ok=True)
    # Train the model
    total_step = len(train_loader)
    torch.save(model.state_dict(), os.path.join("models", exp_name, f"init") + ".pth")
    print("Training model...")
    for epoch in range(param.num_epochs):
        i, loss = train_epoch(criterion, epoch, i, model, optimizer, train_loader, writer)
        print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
              .format(epoch + 1, NUM_EPOCHS_LIST, i + 1, total_step, loss.item()))
        if epoch % 5 == 0:
            torch.save(model.state_dict(), os.path.join("models", exp_name, f"epoch-{epoch}") + ".pth")
        # Validation
        epoch_validation(criterion, epoch, model, valid_loader, writer)
    torch.save(model.state_dict(), os.path.join("models", exp_name, f"epoch-{param.num_epochs}") + ".pth")
    # Test the model
    print("Testing model...")
    test_model(device, model, test_loader, writer)
    writer.flush()
    writer.close()


def test_model(device, model, test_loader, writer):
    with model.eval():
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            writer.add_scalar("Accuracy/test", 100 * correct / total, 0)


def epoch_validation(criterion, epoch, model, valid_loader, writer):
    with torch.no_grad():
        for name, m in model.named_modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d):
                writer.add_histogram(f"bias/{name}", m.bias, epoch)
        correct = 0
        total = 0
        for images, labels in valid_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            writer.add_scalar("Loss/validation", loss, epoch)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            del images, labels, outputs
        writer.add_scalar("Accuracy/validation", 100 * correct / total, epoch)


def train_epoch(criterion, epoch, i, model, optimizer, train_loader, writer):
    loss = None
    for i, (images, labels) in enumerate(train_loader):
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        writer.add_scalar("Loss/train", loss, epoch)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return i, loss


def init_training(device, param):
    print("initializing training...", end=' ')
    train_loader, valid_loader = get_train_valid_loader(data_dir='./data', batch_size=param.batch_size,
                                                        augment=False, random_seed=42)
    test_loader = get_test_loader(data_dir='./data', batch_size=param.batch_size)

    train_loader = DeviceDataLoader(train_loader, device)
    valid_loader = DeviceDataLoader(valid_loader, device)
    test_loader = DeviceDataLoader(test_loader, device)

    torch.manual_seed(42)
    model = AlexNet(**param.to_dict()).to(device)
    if param.reinitialize:
        model.reinitialize(seed=42)
    torch.manual_seed(42)
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=param.lr, momentum=0.9)
    print("Done.")
    return criterion, model, optimizer, test_loader, train_loader, valid_loader


if __name__ == '__main__':
    main()
