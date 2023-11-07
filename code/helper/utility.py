import torch
import pickle
import matplotlib.pyplot as plt
import numpy as np
import random
from torch.utils.data import DataLoader, ConcatDataset, Subset
from sklearn.metrics import confusion_matrix, classification_report
import scikitplot as skplt
from collections import defaultdict, OrderedDict
from torchvision import datasets, transforms
import os
import shutil


def select_devices(use_cudnn_if_avaliable):
    if torch.cuda.is_available():
        # Enable cuDNN auto-tuner
        if use_cudnn_if_avaliable:
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
        print("using CUDA + cudnn")
        return torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        print("using mac mps")
        return torch.device("mps")
    else:
        print("using CPU")
        return torch.device("cpu")


def save_model(model, path):
    torch.save(model.state_dict(), path)


def save_pickle_files(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def read_pickle_files(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def model_validation(model, device, data_loader, pth_path, record_save_path):
    model.load_state_dict(torch.load(pth_path))
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    # for confusion matrix
    y_true = []
    y_pred = []
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)  # predicted is the emotion index
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # for confusion matrix
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy}%")
    # calculate confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 7))
    skplt.metrics.plot_confusion_matrix(y_true, y_pred, figsize=(7, 7), normalize=True)
    plt.savefig(record_save_path + "/confusion_matrix.png")
    print(classification_report(y_true, y_pred))


def plot_record(x, y, xlabel, ylabel, title, save_path, show=True):
    plt.clf()
    plt.plot(x, y)
    # plt.scatter(x, y, s=1)
    lowest_idx = y.index(min(y))
    highest_idx = y.index(max(y))
    plt.scatter(x[lowest_idx], y[lowest_idx], c='red', marker='o', s=10, label='Lowest')
    plt.scatter(x[highest_idx], y[highest_idx], c='blue', marker='o', s=10, label='Highest')
    plt.text(x[lowest_idx], y[lowest_idx], f'X:{x[lowest_idx]}, Y:{y[lowest_idx]:.2f}', 
             fontsize=12, ha='right', va='bottom', color='red')
    plt.text(x[highest_idx], y[highest_idx], f'X:{x[highest_idx]}, Y:{y[highest_idx]:.2f}', 
             fontsize=12, ha='left', va='top', color='blue')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(save_path)
    if show:
        plt.show()


def combine_data(dataset1, dataset2, batch_size, shuffle=False, pin_memory=False, num_workers=0):
    len_dataset1 = len(dataset1)
    len_dataset2 = len(dataset2)
    if len_dataset1 > len_dataset2:
        larger_dataset = dataset1
        smaller_dataset = dataset2
    else:
        larger_dataset = dataset2
        smaller_dataset = dataset1
    num_samples_to_select = len(smaller_dataset)
    selected_samples = random.sample(range(len(larger_dataset)), num_samples_to_select)
    selected_larger_dataset = Subset(larger_dataset, selected_samples)
    combined_dataset = ConcatDataset([smaller_dataset, selected_larger_dataset])
    dataloader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=shuffle, 
                            pin_memory=pin_memory, num_workers=num_workers)
    return dataloader


def imbalance_combine_data(datasets, batch_size, shuffle=False, pin_memory=False, num_workers=0):
    combined_dataset = ConcatDataset(datasets)
    dataloader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=shuffle, 
                            pin_memory=pin_memory, num_workers=num_workers)
    return dataloader


def check_label_size(dataloader):
    label_counts = defaultdict(int)
    for _, labels in dataloader:
        labels = labels.tolist()
        for label in labels:
            label_counts[label] += 1
    for label, count in label_counts.items():
        print(f"Label {label}: {count} samples")
    return label_counts


def balance_weight(train_label_count, train_loader_len, batch_size):
    class_weights = {}
    for class_label, class_count in train_label_count.items():
        class_weights[class_label] = (train_loader_len*batch_size / (len(train_label_count) * class_count))
    class_weights = OrderedDict(class_weights.items())
    print(f"adjusted weights: {class_weights}")
    return class_weights


def grayscale_transform_rotation(channel, size, degree):
    transformer = transforms.Compose([
        transforms.Grayscale(num_output_channels=channel),
        transforms.RandomRotation(degrees=(-degree, degree)),
        transforms.Resize(size), 
        transforms.ToTensor(),
    ])
    return transformer


def grayscale_transform_verticalflip(channel, size, p=0.5):
    transformer = transforms.Compose([
        transforms.Grayscale(num_output_channels=channel),
        transforms.RandomVerticalFlip(p),
        transforms.Resize(size), 
        transforms.ToTensor(),
    ])
    return transformer


def grayscale_transform_horizontalflip(channel, size, p=0.5):
    transformer = transforms.Compose([
        transforms.Grayscale(num_output_channels=channel),
        transforms.RandomHorizontalFlip(p),
        transforms.Resize(size), 
        transforms.ToTensor(),
    ])
    return transformer


def grayscale_transform_crop(channel, size, scale=(0.08, 1)):
    transformer = transforms.Compose([
        transforms.Grayscale(num_output_channels=channel),
        transforms.RandomResizedCrop(size, scale=scale),
        transforms.Resize(size), 
        transforms.ToTensor(),
    ])
    return transformer


def create_augmentation_dataset(source_path, target_path, class_name):
    target_path = '../augmentation_dataset/' + target_path + '/'
    if not os.path.exists(target_path):
        os.mkdir(target_path)
        for class_ in os.listdir(source_path):
            target_class_path = target_path + class_ + '/'
            s = source_path + '/' + class_ + '/'
            if class_name == class_:
                shutil.copytree(s, target_class_path)
            elif not os.path.exists(target_class_path):
                os.makedirs(target_class_path)
                any_file = os.listdir(s)[0]
                shutil.copy(s+any_file, target_class_path)
        print(f'data augmentation dataset [{class_name}] created successfuly at [{target_class_path}]')
    else:
        print(f'data augmentation is already created at [{target_path}]')
        
    
