import datetime
import os

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
# from clusterify import clusterify, finalize
import os
import sys
from utils import *
DIR_PATH = R""





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




def get_params(freeze_bias, num_classes, num_epochs, batch_sizes, learning_rate, b_scales, w_scales, reinitialize_list):
    params = []
    for fb in freeze_bias:
        for nc in num_classes:
            for ne in num_epochs:
                for bs in batch_sizes:
                    for lr in learning_rate:
                        params.append([False, fb, nc, ne, bs, lr, 1, np.sqrt(5), None])
                        for b_scale in b_scales:
                            for w_scale in w_scales:
                                for reinitialize in reinitialize_list:
                                    params.append([True, fb, nc, ne, bs, lr, b_scale, w_scale, reinitialize])
    return params


# %% constants
FREEZE_BIAS_LIST = [False, True]
NUM_CLASSES_LIST = [10]
NUM_EPOCHS_LIST = [20]
BATCH_SIZE_LIST = [64, 128]
LR_LIST = [1e-4, 5e-3, 1e-3]
BIAS_VAR_LIST = [0.1, 1, 5, 10]
WEIGHT_VAR_LIST = [0.1, np.sqrt(5), 1]
REINITIALIZE_LIST = [["l1_conv"], ["l2_conv"], ["l3_conv"], ["l4_conv"], ["l5_conv"],
                     ["l1_batchnorm"], ["l2_batchnorm"], ["l3_batchnorm"], ["l4_batchnorm"], ["l5_batchnorm"],
                     ["fc.fc"], ["fc1.fc1"], ["fc2.fc2"]]


def main():
    params = get_params(FREEZE_BIAS_LIST, NUM_CLASSES_LIST, NUM_EPOCHS_LIST, BATCH_SIZE_LIST, LR_LIST, BIAS_VAR_LIST,
                        WEIGHT_VAR_LIST, REINITIALIZE_LIST)

    params_df = pd.DataFrame(params,
                             columns=["reinitialize", "freeze_bias", "num_classes", "num_epochs", "batch_size", "lr",
                                      "b_scale", "w_scale", "reinitialize_list"])
    os.makedirs(os.path.join(DIR_PATH, "models"), exist_ok=True)
    os.makedirs(os.path.join(DIR_PATH, "runs"), exist_ok=True)
    os.makedirs(os.path.join(DIR_PATH, "data"), exist_ok=True)
    results = []
    for i, param in params_df.iterrows():
        results.append(end_to_end_model_train(i, param))
        if len(sys.argv) > 1 and sys.argv[1] == "test":
            break
    # finalize(results)


# @clusterify(chunk_size=1, n_jobs=1 if (len(sys.argv) > 1 and sys.argv[1] == "test") else 50,
#             job_script_prologue=['module load cuda/12.4.1', 'module load nvidia'],
#             memory='16GB', walltime='1:00:00',
#             job_extra_directives=['--gres=gpu:a30:1', '--job-name=alexnet',
#                                   '--output=/sci/labs/uvhart/odedwer/logs/alexnet-%j.out'])
def end_to_end_model_train(i, param):
    device = get_device()
    criterion, model, optimizer, test_loader, train_loader, valid_loader = init_training(device, param)
    writer, exp_name = get_summary_writer("alexnet", param) #change
    os.makedirs(os.path.join("models", exp_name), exist_ok=True)
    # Train the model
    total_step = len(train_loader)
    torch.save(model.state_dict(), os.path.join("models", exp_name, f"init") + ".pth")
    print("Training model...")
    for epoch in range(param.num_epochs):
        i, loss = train_epoch(criterion, epoch, i, model, optimizer, train_loader, writer)
        print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
              .format(epoch + 1, param.num_epochs, i + 1, total_step, loss.item()))
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
    return None



def init_training(device, param):
    print("initializing training...", end=' ')
    train_loader, valid_loader = get_train_valid_loader(data_dir=os.path.join(DIR_PATH, "data"),
                                                        batch_size=param.batch_size,
                                                        augment=False, random_seed=42)
    test_loader = get_test_loader(data_dir=os.path.join(DIR_PATH, "data"), batch_size=param.batch_size)

    train_loader = DeviceDataLoader(train_loader, device)
    valid_loader = DeviceDataLoader(valid_loader, device)
    test_loader = DeviceDataLoader(test_loader, device)

    torch.manual_seed(42)
    model = AlexNet(**param.to_dict()).to(device)
    if param.reinitialize and param.reinitialize_list is None:
        model.reinitialize(seed=42)
    elif param.reinitialize and param.reinitialize_list:
        for name in param.reinitialize_list:
            if "conv" in name:
                model._reinitialize_conv(model[name])
            elif "fc" in name:
                model._reinitialize_fc(model[name])
            elif "batchnorm" in name:
                model._reinitialize_batchnorm(model[name])
    torch.manual_seed(42)
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=param.lr, momentum=0.9)
    print("Done.")
    return criterion, model, optimizer, test_loader, train_loader, valid_loader


if __name__ == '__main__':
    main()
