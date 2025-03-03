# get train and test dataloaders from the face-expression dataset
import numpy as np
from torchvision.io import read_image

from utils import *
from torch.utils.data import Subset, DataLoader
from sklearn.model_selection import train_test_split

from modules import DeviceDataLoader, FER2013, SimpleCNN
#%%
# calculate the mean and std of the dataset

mean,std = [],[]
for label in os.listdir('data/face-expression/train'):
    for img in os.listdir(f'data/face-expression/train/{label}'):
        img = read_image(f'data/face-expression/train/{label}/{img}')
        img = np.array(img)/255
        mean.append(np.mean(img))
        std.append(np.std(img))


#%%
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
PARAM = pd.Series(
    index=['name', 'num_epochs', 'batch_size', 'lr', 'b_scale', 'w_scale', 'num_classes', 'reinitialize',
           'freeze_bias', 'conv_kwargs', 'pool_kwargs', 'n_blocks_increasing', 'n_block_decreasing'],
    data=['SimpleCNN', 60, 64, 1e-4, 0.1, 2.2, 7, False, False, {'kernel_size': 3, 'stride': 1, 'padding': 1},
          {'kernel_size': 2, 'stride': 2}, 3, 1]
)
samples_weight = np.array([class_weights[int(t)] for t in train_dataset.labels[train_split.indices]])
sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(samples_weight))
train_loader = DeviceDataLoader(DataLoader(train_split, batch_size=PARAM.batch_size, sampler=sampler), device=device)
valid_loader = DeviceDataLoader(DataLoader(val_split, batch_size=PARAM.batch_size), device=device)
test_loader = DeviceDataLoader(DataLoader(test_dataset, batch_size=PARAM.batch_size), device=device)
# %%
end_to_end_model_train(0, PARAM, MODEL, CRITERION, OPTIMIZER, train_loader, valid_loader, test_loader)
