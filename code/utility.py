import torch
import pickle
import matplotlib.pyplot as plt


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


def model_validation(model, device, data_loader, pth_path):
    model.load_state_dict(torch.load(pth_path))
    model.to(device)
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)  # predicted is the emotion index
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy}%")


def plot_record(x, y, xlabel, ylabel, title, save_path):
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(save_path)
    plt.show