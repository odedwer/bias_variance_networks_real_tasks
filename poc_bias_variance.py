import gc
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms

# Assuming your original files are in the same directory
from face_recognition_model_comparison import SimpleCNN
from modules import DeviceDataLoader
from utils import get_summary_writer


class FER2013BinaryDataset(Dataset):
    """A modified dataset class to only load 'happy' and 'sad' classes."""

    def __init__(self, img_dir, transform=None, classes_to_use=['happy', 'sad']):
        self.img_dir = img_dir
        self.transform = transform
        self.images = []
        self.labels = []
        # Create new mappings for binary classification (0: happy, 1: sad)
        self.label_number_map = {label: i for i, label in enumerate(classes_to_use)}
        self.number_label_map = {i: label for i, label in enumerate(classes_to_use)}
        self.unique_labels = classes_to_use

        for label in classes_to_use:
            class_path = os.path.join(img_dir, label)
            if not os.path.isdir(class_path):
                print(f"Warning: Directory not found for class '{label}'")
                continue
            class_images = [os.path.join(class_path, img) for img in os.listdir(class_path)]
            self.images.extend(class_images)
            self.labels.extend([self.label_number_map[label]] * len(class_images))

        self.labels = np.array(self.labels)
        print(
            f"Loaded {len(self.images)} images from {len(classes_to_use)} classes for the {os.path.basename(img_dir)} set.")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx])
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


def run_training_poc(name, init_bias=None):
    torch.backends.cudnn.benchmark = True  # <-- Add this
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 1. Data Preparation ---
    train_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,))
    ])
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,))
    ])

    full_train_dataset = FER2013BinaryDataset('data/face-expression/train', transform=train_transforms)
    test_dataset = FER2013BinaryDataset('data/face-expression/test', transform=test_transforms)

    # Create a stratified train/validation split
    train_indices, val_indices, _, _ = train_test_split(
        list(range(len(full_train_dataset))), full_train_dataset.labels,
        stratify=full_train_dataset.labels,
        test_size=0.2,
        random_state=42
    )
    train_split = Subset(full_train_dataset, train_indices)
    val_split = Subset(full_train_dataset, val_indices)

    train_loader = DeviceDataLoader(
        DataLoader(train_split, batch_size=64, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True),
        device)
    val_loader = DeviceDataLoader(
        DataLoader(val_split, batch_size=64, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True),
        device)
    test_loader = DeviceDataLoader(
        DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True),
        device)

    lr = 1e-3
    num_epochs = 100

    # --- 3. Training Loop ---
    print("\n" + "=" * 20)
    print(f"STARTING EXPERIMENT: {name}")
    print("=" * 20)
    for i, seed in enumerate([42, 3, 97]):
        torch.manual_seed(seed)
        model = SimpleCNN(num_classes=2, bn=False, init_bias=init_bias).to(device)

        param_series = pd.Series({"model": "SimpleCNN_Binary", "bn": True, "init_bias": init_bias})
        writer, log_dir_name = get_summary_writer(f"POC_{name}_seed{i + 1}", param_series)
        os.makedirs(os.path.join("models", log_dir_name), exist_ok=True)

        optimizer = optim.AdamW(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        # --- Add GradScaler ---
        scaler = torch.amp.GradScaler('cuda')
        # save the initial model
        init_model_path = os.path.join("models", log_dir_name, "epoch-0.pth")
        torch.save(model.state_dict(), init_model_path)
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        val_acc = 100.0 * correct / total
        writer.add_scalar("Accuracy/validation", val_acc, 0)
        for epoch in range(num_epochs):
            model.train()
            for images, labels in train_loader:
                # images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()

                # --- Use autocast ---
                with torch.amp.autocast('cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            val_acc = 100.0 * correct / total
            writer.add_scalar("Accuracy/validation", val_acc, epoch + 1)
            model_path = os.path.join("models", log_dir_name, f"epoch-{epoch + 1}.pth")
            torch.save(model.state_dict(), model_path)
            print(f"Epoch {epoch + 1}/{num_epochs}, Val Acc: {val_acc:.2f}%")

        final_model_path = os.path.join("models", log_dir_name, "epoch-final.pth")
        torch.save(model.state_dict(), final_model_path)
        print(f"Finished training {name}. Final model saved.")

        # --- 4. Test Evaluation (ADDED) ---
        print(f"--- Evaluating {name} on the test set ---")
        model.load_state_dict(torch.load(final_model_path))
        model.eval()

        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        test_acc = 100.0 * correct / total
        writer.add_scalar("Accuracy/test", test_acc, 0)
        print(f"✅ Final Test Accuracy for {name}: {test_acc:.2f}%")

        writer.close()


def run_exp_loop():
    # --- 2. Experiment Configurations ---
    experiments = {
        # "No_Variance_Bias": {"init_bias": 0.0},
        "Low_Variance_Bias": {"init_bias": .1},
        # "Low2_Variance_Bias": {"init_bias": 1.0},
        # "High1_Variance_Bias": {"init_bias": 5.0},
        "High_Variance_Bias": {"init_bias": 10.0}
    }
    for exp, config in experiments.items():
        print(f"Experiment: {exp}, Config: {config}")
        run_training_poc(exp, config['init_bias'])
        # clean up GPU memory and force garbage collection
        torch.cuda.empty_cache()

        gc.collect()

    print("\n✅ All experiments have finished.")


if __name__ == '__main__':
    run_exp_loop()
