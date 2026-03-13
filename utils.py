"""
Training utilities for neural networks.

Provides helper functions for:
- Device management and training setup
- Experiment logging and file management
- Model training and validation loops
- Visualization of training metrics
"""

import datetime
import re

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import os
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


def get_device():
    """Get available device (CUDA if available, else CPU)."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device: " + str(device))
    return device


def make_filename_safe(filename):
    """Replace special characters with underscores to create filesystem-safe names."""
    # Replace any character that is not alphanumeric, a space, or a hyphen with an underscore
    safe_filename = re.sub(r'[^.a-zA-Z0-9\s-]+', '_', filename)
    # Replace spaces with underscores
    safe_filename = safe_filename.replace(' ', '')
    return safe_filename


def get_summary_writer(model_name, param, classes=None):
    """
    Create TensorBoard SummaryWriter and experiment directory.
    
    Args:
        model_name (str): Name of the model
        param (pd.Series): Training parameters to save
        classes (list, optional): List of class names
        
    Returns:
        tuple: (SummaryWriter, experiment_name)
    """
    timestamp = str(datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S"))
    # save parameters to a file in the experiment directory
    classes_string = "-".join(classes) if classes else ""
    exp_path = os.path.join("runs", timestamp + "_" + model_name + "_" + classes_string).replace("\\", "/")
    os.makedirs(exp_path, exist_ok=True)

    param.to_csv(os.path.join(exp_path, "params.csv").replace("\\", "/"))

    return SummaryWriter(log_dir=exp_path), exp_path[exp_path.find("/") + 1:]


def init_training(device, model, param, criterion, optimizer, optimizer_kwargs):
    """
    Initialize model, optimizer, criterion and apply custom reinitialization.
    
    Args:
        device: torch device
        model: Model class (not instantiated)
        param (pd.Series): Training parameters including custom bias/weight scaling
        criterion: Loss function class
        optimizer: Optimizer class
        optimizer_kwargs (dict): Additional kwargs for optimizer
        
    Returns:
        tuple: (model, criterion, optimizer)
    """
    m.to(device)
    optimizer = optimizer(filter(lambda p: p.requires_grad, m.parameters()), lr=param.get('lr', 1e-4),
                          **optimizer_kwargs)
    criterion = criterion()
    torch.manual_seed(42)
    if param.get('reinitialize', None) and not param.get("reinitialize_list", None):
        m.reinitialize()
    elif param.get('reinitialize', None) and param.get("reinitialize_list", None):
        m.partial_reinitialize(param.reinitialize_list)
    torch.manual_seed(42)
    return m, criterion, optimizer


def train_epoch(criterion, epoch, i, model, optimizer, train_loader, writer):
    """
    Execute one training epoch.
    
    Args:
        criterion: Loss function
        epoch (int): Current epoch number
        i (int): Step counter
        model: Neural network model
        optimizer: Optimizer
        train_loader: DataLoader for training data
        writer: TensorBoard SummaryWriter
        
    Returns:
        tuple: (final_step_count, final_loss)
    """
    # Forward pass
    outputs = model(images)
    loss = criterion(outputs, labels)
    writer.add_scalar("Loss/train", loss, epoch)

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return i, loss


def plot_confusion_matrix(y_true, y_pred, dataset, writer, epoch, extra_name=""):
    """
    Plot and log confusion matrix to TensorBoard.
    
    Args:
        y_true (array-like): True labels
        y_pred (array-like): Predicted labels
        dataset: Dataset object with label mappings
        writer: TensorBoard SummaryWriter
        epoch (int): Epoch number for logging
        extra_name (str): Prefix for the figure name
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm = cm.astype('float') / cm.sum(axis=0, keepdims=True)
    df_cm = pd.DataFrame(cm, index=labels, columns=labels)
    df_cm.rename(columns=dataset.number_label_map, index=dataset.number_label_map, inplace=True)
    plt.figure(figsize=(10, 7))
    sns.heatmap(df_cm, annot=True, cbar=False, fmt=".2f", cmap='jet')
    plt.xlabel("True")
    plt.ylabel("Predicted")
    writer.add_figure(extra_name + "Confusion Matrix", plt.gcf(), epoch)
    plt.close()


def epoch_validation(criterion, epoch, model, valid_loader, writer):
    """
    Validate model on validation set and log metrics.
    
    Logs accuracy, bias histograms, and confusion matrices to TensorBoard.
    
    Args:
        criterion: Loss function
        epoch (int): Epoch number
        model: Neural network model
        valid_loader: Validation DataLoader
        writer: TensorBoard SummaryWriter
    """
    with torch.no_grad():
        for name, m in model.named_modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d):
                writer.add_histogram(f"bias/{name}", m.bias, epoch)
        correct = 0
        total = 0
        y_true = []
        y_pred = []
        for images, labels in valid_loader:
            outputs = model(images)
            # add validation loss to tensorboard
            loss = criterion(outputs, labels)
            writer.add_scalar("Loss/validation", loss, epoch)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            del images, labels, outputs
        writer.add_scalar("Accuracy/validation", 100 * correct / total, epoch)
        # plot confusion matrix
        plot_confusion_matrix(y_true, y_pred, valid_loader.dl.dataset.dataset, writer, epoch)


def test_model(device, model, test_loader, writer):
    """
    Evaluate model on test set and log metrics.
    
    Args:
        device: torch device
        model: Neural network model
        test_loader: Test DataLoader
        writer: TensorBoard SummaryWriter
    """
    with torch.no_grad():
        correct = 0
        total = 0
        y_true = []
        y_pred = []
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            writer.add_scalar("Accuracy/test", 100 * correct / total, 0)
        # plot confusion matrix
        plot_confusion_matrix(y_true, y_pred, test_loader.dl.dataset, writer, 0, "test/")


def end_to_end_model_train(i, param, model, criterion, optimizer, train_loader, valid_loader, test_loader,
                           optimizer_kwargs=None):
    """
    Complete training pipeline: initialize, train, validate, and test model.
    
    Saves model checkpoints and logs all metrics to TensorBoard.
    
    Args:
        i (int): Step counter
        param (pd.Series): Training parameters
        model: Model class
        criterion: Loss function class
        optimizer: Optimizer class
        train_loader: Training DataLoader
        valid_loader: Validation DataLoader
        test_loader: Test DataLoader
        optimizer_kwargs (dict, optional): Additional optimizer arguments
    """
    device = get_device()
    model, criterion, optimizer = init_training(device, model, param, criterion, optimizer, (optimizer_kwargs or {}))
    writer, exp_name = get_summary_writer(model.__dict__.get("name", "network"), param)
    os.makedirs(os.path.join("models", exp_name), exist_ok=True)
    # Train the model
    total_step = len(train_loader)
    torch.save(model.state_dict(), os.path.join("models", exp_name, f"init") + ".pth")
    print("Training model...")
    for epoch in range(param.num_epochs):
        i, loss = train_epoch(criterion, epoch, i, model, optimizer, train_loader, writer)
        print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
              .format(epoch + 1, param.num_epochs, i + 1, total_step, loss.item()))
        if epoch % 20 == 0:
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


def calculate_conv_width_height(width_in, height_in, kernel_size, stride, padding):
    """
    Calculate output spatial dimensions after convolution and pooling.
    
    Args:
        width_in (int): Input width
        height_in (int): Input height
        kernel_size (int): Convolution kernel size
        stride (int): Convolution stride
        padding (int): Convolution padding
        
    Returns:
        tuple: (output_width, output_height)
    """
    width_out, height_out = ((width_in + 2 * padding - kernel_size) // stride) + 1, (
            (height_in + 2 * padding - kernel_size) // stride) + 1
    width_out = ((width_out - kernel_size) // 2) + 1
    height_out = ((height_out - kernel_size) // 2) + 1
    return width_out + 1, height_out + 1
