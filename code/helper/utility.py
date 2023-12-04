import torch
import pickle
import matplotlib.pyplot as plt
import numpy as np
import random
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset, Subset
from sklearn.metrics import confusion_matrix, classification_report
import scikitplot as skplt
from collections import defaultdict, OrderedDict
from torchvision import datasets, transforms
import os
import shutil
import cv2


def select_devices(use_cudnn_if_avaliable):
    if torch.cuda.is_available():
        # Enable cuDNN auto-tuner
        if use_cudnn_if_avaliable:
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
            print("using CUDA + cudnn")
        else:
            print("using CUDA")
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


def model_validation(model, device, data_loader, pth_path, record_save_path, file_name, output_softmax=False, show=True, size=7, ext=True):
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
            if output_softmax:
                outputs = nn.functional.softmax(outputs, dim=1)
            predicted = torch.argmax(outputs.data, 1)  # predicted is the emotion index
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # for confusion matrix
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy}%")
    # calculate confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)
    if show:
        plt.figure(figsize=(size, size))
    skplt.metrics.plot_confusion_matrix(y_true, y_pred, figsize=(size, size), normalize=True)
    if ext:
        plt.savefig(record_save_path + "/confusion_matrix" + file_name + f'_{accuracy:{.5}}%' + ".png")
    else:
        plt.savefig(record_save_path + "/confusion_matrix" + file_name + ".png")
    print(classification_report(y_true, y_pred))
    return (correct / total)


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
    label_counts = dict(sorted(label_counts.items()))
    for label, count in label_counts.items():
        print(f"Label {label}: {count} samples")
    return label_counts


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

def grayscale_transform_trainslate(channel, size, value=(0.3, 0)):
    transformer = transforms.Compose([
        transforms.Grayscale(num_output_channels=channel),
        transforms.RandomAffine(0, translate=value),
        transforms.Resize(size), 
        transforms.ToTensor(),
    ])
    return transformer

def grayscale_transform_shear(channel, size, value=(-45, 45)):
    transformer = transforms.Compose([
        transforms.Grayscale(num_output_channels=channel),
        transforms.RandomAffine(0, shear=value),
        transforms.Resize(size), 
        transforms.ToTensor(),
    ])
    return transformer


# create a copy of class in list class_name, use for augmentation transform later
# store in '../augmentation_dataset'
def create_augmentation_dataset(source_path, target_path, class_name):
    target_path = os.path.join('../augmentation_dataset', target_path)
    if not os.path.exists(target_path):
        os.mkdir(target_path)
        for class_ in os.listdir(source_path):
            target_class_path = os.path.join(target_path, class_)
            s = os.path.join(source_path, class_)
            # copy entire folder if class name is in list
            if class_ in class_name:
                shutil.copytree(s, target_class_path)
            # to use as input for dataloader, 
            # other class not in class_name need to contain at least 1 image
            elif not os.path.exists(target_class_path):
                os.makedirs(target_class_path)
                any_file = os.listdir(s)[0]
                any_file_path = os.path.join(s, any_file)
                shutil.copy(any_file_path, target_class_path)
        print(f'data augmentation dataset [{class_}] created successfuly at [{target_path}]')
    else:
        print(f'data augmentation is already created at [{target_path}]')


# split dataset by ratio, use to split training set to trainingset and validation set
def split_dataset_by_ratio(source_path, target_path, ratio=0.8):
    if not os.path.exists(target_path):
        os.mkdir(target_path)
        train = os.path.join(target_path, 'train_data')
        val = os.path.join(target_path, 'validation_data')
        os.mkdir(train)
        os.mkdir(val)
        for class_folder in os.listdir(source_path):
            class_folder_path = os.path.join(source_path, class_folder)
            images = os.listdir(class_folder_path)
            random.shuffle(images)
            split_index = int(len(images) * ratio)
            train_images = images[: split_index]
            val_images = images[split_index:]
            
            train_class_path = os.path.join(train, class_folder)
            os.mkdir(train_class_path)
            val_class_path = os.path.join(val, class_folder)
            os.mkdir(val_class_path)
            
            for image in train_images:
                img_source_path = os.path.join(class_folder_path, image)
                img_target_path = os.path.join(train_class_path, image)
                shutil.copy(img_source_path, img_target_path)
            
            for image in val_images:
                img_source_path = os.path.join(class_folder_path, image)
                img_target_path = os.path.join(val_class_path, image)
                shutil.copy(img_source_path, img_target_path)
    else:
        print(f'Target path [{target_path}] is already existed.')


# Function to add gaussian noise to an image
def add_gaussian_noise(image, mean=0, sigma=25):
    row, col, ch = image.shape
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    noisy = np.clip(image + gauss, 0, 255)
    return noisy.astype(np.uint8)

# Function to add salt-and-pepper noise to an image
def add_salt_and_pepper_noise(image, salt_prob=0.02, pepper_prob=0.02):
    row, col, ch = image.shape
    noisy = np.copy(image)
    # Add salt noise
    salt_pixels = np.random.rand(row, col, ch) < salt_prob
    noisy[salt_pixels] = 255
    # Add pepper noise
    pepper_pixels = np.random.rand(row, col, ch) < pepper_prob
    noisy[pepper_pixels] = 0
    return noisy.astype(np.uint8)
        
# function to add noise to image, will not override if the output path is existed.
# type: 0-gaussian, 1-pepper+salt, 3-mix with gaussian and pepper-salt
# prob: the probability to apply noise to an image
def add_noise(input_path, output_path, target_classes, prob, type):
    total = 0
    generate = 0
    if not os.path.exists(output_path):
        os.mkdir(output_path)
        # Iterate through the images in the input folder
        for classfoldername in os.listdir(input_path):
            class_path = os.path.join(input_path, classfoldername)
            for filename in os.listdir(class_path):
                total += 1
                image_path = os.path.join(class_path, filename)
                original_image = cv2.imread(image_path)
                # Add noise to the image
                noisy_image = None
                if type == 0:
                    noisy_image = add_gaussian_noise(original_image)
                elif type == 1:
                    noisy_image = add_salt_and_pepper_noise(original_image)
                elif type == 2:
                    select = random.random()
                    if select > 0.5:
                        noisy_image = add_gaussian_noise(original_image)
                    else:
                        noisy_image = add_salt_and_pepper_noise(original_image)
                # Save the noisy image to the output folder
                output_class_path = os.path.join(output_path, classfoldername)
                if not os.path.exists(output_class_path):
                    os.mkdir(output_class_path)
                r = random.random()
                if r > prob and (classfoldername in target_classes):
                    generate += 1
                    image_output_path = os.path.join(output_class_path,f"noisy_{filename}")
                    cv2.imwrite(image_output_path, noisy_image)
                else:
                    image_output_path = os.path.join(output_class_path,f"{filename}")
                    cv2.imwrite(image_output_path, original_image)
        print(f'Total: {total}, generate: {generate}')
    else:
        print(f'folder [{output_path}] is already existed')
