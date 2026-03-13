import pandas as pd
import numpy as np
from PIL import Image
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
import os
#import kagglehub

# Download latest version
#path = kagglehub.dataset_download("msambare/fer2013")

from ResNet import ResNet
from utils import get_summary_writer, plot_confusion_matrix

class FER2013Dataset(Dataset):
    def __init__(self, img_dir, transform=None, classes=None):
        self.img_dir = img_dir
        if not classes:
            self.unique_labels = os.listdir(img_dir)
        else:
            self.unique_labels = classes
        self._img_count = []
        self.label_number_map = {}
        self.number_label_map = {}
        for i, label in enumerate(self.unique_labels):
            self._img_count.append(len(os.listdir(os.path.join(img_dir, label))))
            self.number_label_map[i] = label
            self.label_number_map[label] = i
        self.labels = []
        for l, c in zip(self.unique_labels, self._img_count):
            self.labels.extend([self.label_number_map[l]] * c)
            # self.labels.extend([l] * c)

        self.images = []
        for label in self.unique_labels:
            self.images.extend([os.path.join(self.img_dir, label, img) for img in
                                os.listdir(os.path.join(img_dir, label))])
        self.transform = transform
        self.labels = np.array(self.labels)

    def __len__(self):
        return sum(self._img_count)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx])
        if self.transform:
            img = self.transform(img)

        return img, self.labels[idx]

    def get_class_weights(self):
        return 1 / np.array(self._img_count)


train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),  # random rotation ±10 degrees
    transforms.RandomCrop(48, padding=4),  # random crop with padding
    transforms.ToTensor(), 
    transforms.Normalize(mean=(0.5,), std=(0.5,))  # normalize to [-1,1] roughly
])
# For validation/test, we usually avoid random transforms, just normalize
test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,))
])

classes=["fear","angry"]
train_dataset = FER2013Dataset('data/face-expression/train', transform=train_transforms, classes=classes)
train_indices, validation_indices, _, _ = train_test_split(
    list(range(len(train_dataset))), train_dataset.labels,
    stratify=train_dataset.labels,
    test_size=0.2,
    random_state=42
)
train_split = Subset(train_dataset, train_indices)
val_split = Subset(train_dataset, validation_indices)

test_dataset = FER2013Dataset('data/face-expression/test', transform=test_transforms, classes=classes)
class_weights = train_dataset.get_class_weights()
# samples_weight = np.array([class_weights[int(t)] for t in train_dataset.labels[train_split.indices]])

train_loader = DataLoader(train_split, batch_size=64, shuffle=True, num_workers=4)
val_loader = DataLoader(val_split, batch_size=64, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)


# %%
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2, bn=False, init_bias=None):
        super(SimpleCNN, self).__init__()
        # Convolutional layers
        self.use_bn = bn
        # First conv: 1 input channel (grayscale), 32 filters
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=not bn)
        self.bn1 = nn.BatchNorm2d(32) if bn else nn.Identity()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=not bn)
        self.bn2 = nn.BatchNorm2d(64) if bn else nn.Identity()
        # Max-pooling will reduce spatial size
        self.pool = nn.MaxPool2d(2, 2)
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 12 * 12, 128)  # 48x48 -> pooled to 24x24 -> pooled to 12x12
        self.fc2 = nn.Linear(128, num_classes)
        # Initialize biases if init_bias is given (for conv and fc layers)
        if init_bias is not None:
            init_module_bias(self, init_bias)

    def forward(self, x):
        # Two conv/pool layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)
        # Flatten
        x = x.view(x.size(0), -1)
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


from torchvision import models

# If we want a ResNet without BN, we can disable or remove BN layers.
# One simple way: replace all nn.BatchNorm2d layers with Identity.
import types


def remove_bn(module):
    for name, child in list(module.named_children()):
        if isinstance(child, nn.BatchNorm2d):
            setattr(module, name, nn.Identity())
        else:
            # recurse for nested modules (ResNet has BN inside layers)
            remove_bn(child)


def init_module_bias(module, init_bias=0.0):
    for name, child in module.named_children():
        if isinstance(child, nn.Conv2d) or isinstance(child, nn.Linear) or isinstance(child, nn.BatchNorm2d):
            if child.bias is not None:
                if init_bias == 0.0:
                    nn.init.zeros_(child.bias)
                else:
                    nn.init.uniform_(child.bias, -init_bias, init_bias)

def get_resnet(bn=True, init_bias=None, num_classes=2):
    resnet_model = ResNet(bn=bn, bias=init_bias is not None, num_classes=num_classes)
    if init_bias is not None:
        init_module_bias(resnet_model, init_bias)
    return resnet_model


# %%

import torch.optim as optim


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lr = 1e-3
    num_epochs = 70
    bn_list = [False]
    init_bias_list = [0.0, 10.0, None, 1.0]
    seeds =  [42, 0, 1, 143, 98,  
             11, 7, 13, 21, 27, 31, 37, 43, 49, 53, 59,
             61, 67, 71, 73, 79, 83, 89, 97, 101, 103,
             107, 109, 113, 127]
    model_list, titles, params = get_models(lr, num_epochs, bn_list, init_bias_list, seeds=seeds, resnet=True, simple=True)
    train_models(model_list, titles, params, device, lr, num_epochs)

def set_deterministic(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def get_models(lr, num_epochs, bn_list, init_bias_list, seeds=[42], resnet=True, simple=True
               ):
    model_list = []
    titles = []
    params = []
    for comb in [(bn, init_bias, seed) for bn in bn_list for init_bias in init_bias_list for seed in seeds]:
        if simple:
            set_deterministic(comb[2])
            model_list.append(SimpleCNN(bn=comb[0], init_bias=comb[1], num_classes=2))
            params.append({"model": "SimpleCNN", "bn": comb[0], "init_bias": comb[1], "lr": lr, "num_epochs": num_epochs})
            titles.append(f"SimpleCNN, BN={comb[0]}, Bias={comb[1]}, seed={comb[2]}")
        if resnet:
            set_deterministic(comb[2])
            model_list.append(get_resnet(bn=comb[0], init_bias=comb[1], num_classes=2))
            params.append({"model": "resnet18", "bn": comb[0], "init_bias": comb[1], "lr": lr, "num_epochs": num_epochs})
            titles.append(f"ResNet, BN={comb[0]}, Bias={comb[1]}, seed={comb[2]}, cuda_seed")
    return model_list, titles, params


def train_models(model_list, titles, params, device, lr, num_epochs):
    for model, name, param in zip(model_list, titles, params):
        # Define optimizer and loss function
        model = model.to(device)
        writer, exp_name = get_summary_writer(model.__dict__.get("name", name), pd.Series(param), classes=classes)
        os.makedirs(os.path.join("models", exp_name), exist_ok=True)
        # Train the model
        torch.save(model.state_dict(), os.path.join("models", exp_name, f"init") + ".pth")
        print("\n\n" + "=" * 20)
        print(f"Training model {name}...")
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(weight=torch.Tensor(class_weights).to(device))
        scaler = torch.amp.GradScaler('cuda')
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                # Forward pass
                with torch.autocast(device_type="cuda"):
                    output = model(images)
                    loss = criterion(output, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                running_loss += loss.item() * images.size(0)
                writer.add_scalar("Loss/train", loss, epoch)
                del images, labels, output, loss
            epoch_loss = running_loss / len(train_loader.dataset)
            if epoch % 20 == 0:
                torch.save(model.state_dict(), os.path.join("models", exp_name, f"epoch-{epoch}") + ".pth")

            # Validation accuracy computation
            model.eval()
            for name, m in model.named_modules():
                if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d):
                    if m.bias is not None:
                        writer.add_histogram(f"bias/{name}", m.bias, epoch)
            correct = 0
            total = 0
            all_preds = []
            all_labels = []
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    all_preds.extend(labels.cpu().numpy())
                    all_labels.extend(predicted.cpu().numpy())
                    del images, labels
            val_acc = 100.0 * correct / total
            writer.add_scalar("Accuracy/validation", 100 * correct / total, epoch)
            # plot confusion matrix
            plot_confusion_matrix(all_labels, all_preds, val_loader.dataset.dataset, writer, epoch)
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Val Acc: {val_acc:.2f}%")
        torch.save(model.state_dict(), os.path.join("models", exp_name, f"epoch-final") + ".pth")
        del model


if __name__ == '__main__':
    main()
