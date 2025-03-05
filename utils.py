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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device: " + str(device))
    return device


def make_filename_safe(filename):
    # Replace any character that is not alphanumeric, a space, or a hyphen with an underscore
    safe_filename = re.sub(r'[^.a-zA-Z0-9\s-]+', '_', filename)
    # Replace spaces with underscores
    safe_filename = safe_filename.replace(' ', '')
    return safe_filename


def get_summary_writer(model_name, param):
    timestamp = str(datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S"))
    # save parameters to a file in the experiment directory
    exp_path = os.path.join("runs", timestamp + "_" + model_name).replace("\\", "/")
    os.makedirs(exp_path, exist_ok=True)

    param.to_csv(os.path.join(exp_path, "params.csv").replace("\\", "/"))

    return SummaryWriter(log_dir=exp_path), exp_path[exp_path.find("/") + 1:]


def init_training(device, model, param, criterion, optimizer, optimizer_kwargs):
    m = model(**param.to_dict())
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


def plot_confusion_matrix(y_true, y_pred, dataset, writer, epoch, extra_name=""):
    labels = list(range(len(dataset.unique_labels)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    df_cm = pd.DataFrame(cm, index=labels, columns=labels)
    df_cm.rename(columns=dataset.number_label_map, index=dataset.number_label_map, inplace=True)
    plt.figure(figsize=(10, 7))
    sns.heatmap(df_cm, annot=True, cbar=False, fmt=".2f", cmap='jet')
    plt.xlabel("True")
    plt.ylabel("Predicted")
    writer.add_figure(extra_name + "Confusion Matrix", plt.gcf(), epoch)
    plt.close()


def epoch_validation(criterion, epoch, model, valid_loader, writer):
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
    width_out, height_out = ((width_in + 2 * padding - kernel_size) // stride) + 1, (
            (height_in + 2 * padding - kernel_size) // stride) + 1
    width_out = ((width_out - kernel_size) // 2) + 1
    height_out = ((height_out - kernel_size) // 2) + 1
    return width_out + 1, height_out + 1
