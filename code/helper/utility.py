import torch
import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import scikitplot as skplt  # run "pip3 install scikit-plot" in your env first

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


def plot_record(x, y, xlabel, ylabel, title, save_path):
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
    plt.show()