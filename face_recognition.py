# get train and test dataloaders from the face-expression dataset
from itertools import product

import numpy as np
from torchvision.io import read_image

from utils import *
from torch.utils.data import Subset, DataLoader
from sklearn.model_selection import train_test_split

from modules import DeviceDataLoader, FER2013, SimpleCNN
import traceback

# %%
# calculate the mean and std of the dataset

mean, std = [], []
for label in os.listdir('data/face-expression/train'):
    for img in os.listdir(f'data/face-expression/train/{label}'):
        img = read_image(f'data/face-expression/train/{label}/{img}')
        img = np.array(img) / 255
        mean.append(np.mean(img))
        std.append(np.std(img))

# %%
train_dataset = FER2013('data/face-expression/train')
test_dataset = FER2013('data/face-expression/test')

class_weights = train_dataset.get_class_weights()

device = get_device()

train_indices, validation_indices, _, _ = train_test_split(
    list(range(len(train_dataset))), train_dataset.labels,
    stratify=train_dataset.labels,
    test_size=0.2,
)

train_split = Subset(train_dataset, train_indices)
val_split = Subset(train_dataset, validation_indices)

# %%
MODEL = SimpleCNN
CRITERION = nn.CrossEntropyLoss
OPTIMIZER = torch.optim.AdamW

NAME_LIST = ['SimpleCnn']
NUM_EPOCHS_LIST = [60]
BATCH_SIZE_LIST = [64]
LR_LIST = [1e-4, 1e-3]
B_SCALE_LIST = [0.1, 0.5, 1, 5, 10]
W_SCALE_LIST = [1, np.sqrt(2), 2.2, 5]
NUM_CLASSES_LIST = [7]
REINITIALIZE_LIST = [False, True]
FREEZE_BIAS_LIST = [False, True]
CONV_KWARGS_LIST = [
    {'kernel_size': 3, 'stride': 1, 'padding': 1},
    {'kernel_size': 3, 'stride': 2, 'padding': 1},
    {'kernel_size': 5, 'stride': 1, 'padding': 1},
    {'kernel_size': 5, 'stride': 2, 'padding': 1},
    {'kernel_size': 5, 'stride': 2, 'padding': 2},
]
POOL_KWARGS_LIST = [
    {'kernel_size': 2, 'stride': 2}
]
N_BLOCKS_INCREASING_LIST = [1, 2, 3, 4]
N_BLOCKS_DECREASING_LIST = [1, 2]

samples_weight = np.array([class_weights[int(t)] for t in train_dataset.labels[train_split.indices]])

for i, comb in enumerate(product(
        NAME_LIST, NUM_EPOCHS_LIST, BATCH_SIZE_LIST, LR_LIST, B_SCALE_LIST, W_SCALE_LIST, NUM_CLASSES_LIST,
        REINITIALIZE_LIST, FREEZE_BIAS_LIST, CONV_KWARGS_LIST, POOL_KWARGS_LIST, N_BLOCKS_INCREASING_LIST,
        N_BLOCKS_DECREASING_LIST
)):
    param = pd.Series(
        index=['name', 'num_epochs', 'batch_size', 'lr', 'b_scale', 'w_scale', 'num_classes', 'reinitialize',
               'freeze_bias', 'conv_kwargs', 'pool_kwargs', 'n_blocks_increasing', 'n_block_decreasing'],
        data=comb
    )
    sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(samples_weight))

    train_loader = DeviceDataLoader(DataLoader(train_split, batch_size=param.batch_size, sampler=sampler),
                                    device=device)
    valid_loader = DeviceDataLoader(DataLoader(val_split, batch_size=param.batch_size), device=device)
    test_loader = DeviceDataLoader(DataLoader(test_dataset, batch_size=param.batch_size), device=device)
    try:
        end_to_end_model_train(i, param, MODEL, CRITERION, OPTIMIZER, train_loader, valid_loader, test_loader)
    except Exception as e:
        print(f"Error in training loop {i}")
        print(e)
        traceback.print_exc()
        continue
