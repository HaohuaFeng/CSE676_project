import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle
from torch.utils.data import random_split
import models.optimizer.optimizer as optimizer
from models import VGG16_4096 as m
from helper.training_early_stop import EarlyStop
import helper.utility as utility
import os
import multiprocessing
import platform

# used for multi-thread
os_name = platform.system()
if os_name == "Windows":
    multiprocessing.set_start_method('spawn', force=True)
elif os_name == "Linux":
    multiprocessing.set_start_method("fork")

if __name__ == '__main__':
        
    # optimizer list
    optimizer_configs = ['Adam', 'Adam_amsgrad']

    for optimizer_name in optimizer_configs:

        '''
        0. Data Pre-Processing
        '''
        data_transforms = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        # train_dataset = datasets.ImageFolder(
        #     './dataset/train', transform=data_transforms)
        # split training set to training set and validation set
        # a random seed to ensure reproducibility of results.
        # torch.manual_seed(42)
        # train_size = int(0.8 * len(train_dataset))
        # val_size = len(train_dataset) - train_size
        # train_dataset, val_dataset = random_split(
        #     train_dataset, [train_size, val_size])

        # test_dataset = datasets.ImageFolder(
        #     './dataset/test', transform=data_transforms)


        # train_loader = DataLoader(train_dataset, batch_size=64,
        #                         shuffle=True, num_workers=16, pin_memory=True)
        # val_loader = DataLoader(val_dataset, batch_size=64,
        #                         shuffle=False, num_workers=16, pin_memory=True)
        # test_loader = DataLoader(test_dataset, batch_size=32,
        #                         shuffle=False, num_workers=16, pin_memory=False)
        
        train_dataset = datasets.ImageFolder('../dataset/splited_data/train_data', transform=data_transforms)
        val_dataset = datasets.ImageFolder('../dataset/splited_data/validation_data', transform=data_transforms)
        test_dataset = datasets.ImageFolder('../dataset/test', transform=data_transforms)
        train_loader = DataLoader(train_dataset, batch_size=32,shuffle=True, num_workers=16, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=32,shuffle=False, num_workers=16, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=32,shuffle=False, num_workers=16, pin_memory=False)

        # select device
        device = utility.select_devices(use_cudnn_if_avaliable=True)

        '''
        1. Model
        '''
        # average loss / epoch
        loss_history_per_epoch = []
        # correct prediction / epoch
        correct_prediction_pre_epoch = []
        # accuracy / epoch
        accuracy_per_epoch = []
        # validation loss
        val_loss_per_epoch = []
        # validation accuracy
        val_accuracy_per_epoch = []

        # todo: select optimizer
        # optimizer_name = "Adam_amsgrad"
        # optimizer_name = "Adam"
        test_version = ''
        m.update_file_name(optimizer_name + test_version)
        print('='*20 + '\n' + m.model_name + optimizer_name + test_version)
        
        # initialize model, loss-function and optimizer
        model = m.EmotionCNN(num_classes=7)  # FER-2013 has 7 emotion class
        if not os.path.exists(m.record_save_path):
            os.makedirs(m.record_save_path)
        criterion = nn.CrossEntropyLoss()
        optimizer_ = optimizer.create_optimizer(model.parameters(), optimizer_name)

        # training model
        num_epochs = 200

        # early stopping variables
        stop_counter = 10  # number of count to trigger early stop
        stop_counter_window = stop_counter + 5  # a range to check stop_counter
        different = 0.0001  # different between the best val loss and the most recent one
        different_loss = 0.0001
        stop_counter_interval = 30  # check for early stop for every stop_counter_interval
        counter = 0  # number of count for every trail of early stop
        is_always = True  # always check for early stop, set to true will ignore other setting except stop_counter
        is_exe = False  # is early stop running
        run_after = 0
        early_stopping = EarlyStop(
            m.pth_save_path, m.pth_save_path_loss, stop_counter, different, different_loss, type="accuracy")

        model.to(device)

        # progress bar
        process = tqdm(range(num_epochs),
                    bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}', colour='green', unit='epoch')

        for epoch in process:
            running_loss = 0.0
            accuracy = 0.0
            model.train()
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                # forwarding get output
                outputs = model(inputs)
                # compute loss of output
                loss = criterion(outputs, labels)
                # backward propagation
                optimizer_.zero_grad()
                loss.backward()
                optimizer_.step()
                # record training status
                running_loss += loss.item()
                probability = torch.nn.functional.softmax(outputs, dim=1)
                max_probability, prediction = torch.max(probability, dim=1)
                num_correct_prediction = (prediction == labels).sum().item()
                correct_prediction_pre_epoch.append(num_correct_prediction)
                accuracy += num_correct_prediction / inputs.shape[0]
            # save training status
            loss_history_per_epoch.append((running_loss / len(train_loader)))
            accuracy_per_epoch.append((accuracy / len(train_loader)))

            # training validation + early stopping
            if epoch >= run_after and (is_always or is_exe or epoch % stop_counter_interval == 0):
                val_loss = 0.0
                val_accuracy = 0.0

                if not is_always and epoch % stop_counter_interval == 0:
                    early_stopping.counter = 0
                    is_exe = True

                counter += 1

                if not is_always and counter >= stop_counter_window:
                    counter = 0
                    is_exe = False
                model.eval()
                with torch.no_grad():
                    for inputs, labels in val_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        val_loss += loss.item()
                        probability = torch.nn.functional.softmax(outputs, dim=1)
                        max_probability, prediction = torch.max(probability, dim=1)
                        num_correct_prediction = (prediction == labels).sum().item()
                        accuracy = num_correct_prediction / inputs.shape[0]
                        val_accuracy += accuracy
                val_loss = val_loss / len(val_loader)
                val_accuracy = val_accuracy / len(val_loader)
                val_loss_per_epoch.append(val_loss)
                val_accuracy_per_epoch.append(val_accuracy)

                early_stopping.check_status(model, val_accuracy, val_loss)

                # display recently 5 average loss of epochs
                process.set_description(f"loss= {'{:.5f}'.format(loss_history_per_epoch[-1])} - "
                                            f"val loss= {'{:.5f}'.format(val_loss_per_epoch[-1])} - "
                                            f"accuracy= {'{:.3%}'.format(accuracy_per_epoch[-1])} - "
                                            f"val accuracy= {'{:.3%}'.format(val_accuracy_per_epoch[-1])} - "
                                            f"best= {'{:.3%}'.format(early_stopping.best_of_all_value)} - "
                                            f"Counter= {early_stopping.counter}/{stop_counter}")
            else:
                process.set_description(f"loss= {'{:.5f}'.format(loss_history_per_epoch[-1])} - "
                                            f"accuracy= {'{:.3%}'.format(accuracy_per_epoch[-1])}")

            # if early_stopping.early_stop:
            #     print('\nTrigger Early Stopping\n')
            #     early_stopping.early_stop = False
            #     break

        '''
        2. Save model and records
        '''
        # save the pth file
        torch.save(model.state_dict(), m.pth_manual_save_path)

        utility.save_pickle_files(loss_history_per_epoch,
                                m.record_save_path + '/loss_history.pkl')
        utility.save_pickle_files(
            accuracy_per_epoch, m.record_save_path + '/accuracy_history.pkl')
        utility.save_pickle_files(
            val_loss_per_epoch, m.record_save_path + '/val_loss_history.pkl')
        utility.save_pickle_files(val_accuracy_per_epoch,
                                m.record_save_path + '/val_accuracy_history.pkl')

        '''
        3. Plot records and Evaluation
        '''
        # draw graphs
        data = utility.read_pickle_files(m.record_save_path + '/loss_history.pkl')
        utility.plot_record(x=range(1, len(data)+1), y=data, xlabel="epoch", ylabel="loss", title="Training Loss",
                            save_path=m.record_save_path+"/loss_history.png", show=False)

        data = utility.read_pickle_files(m.record_save_path + '/accuracy_history.pkl')
        utility.plot_record(x=range(1, len(data)+1), y=data, xlabel="epoch", ylabel="accuracy", title="Training Accuracy",
                            save_path=m.record_save_path+"/accuracy_history.png", show=False)

        data = utility.read_pickle_files(m.record_save_path + '/val_loss_history.pkl')
        utility.plot_record(x=range(run_after+1, run_after+len(data)+1), y=data, xlabel="epoch", ylabel="validation loss",
                            title="Validation Loss", save_path=m.record_save_path+"/val_loss_history.png", show=False)

        data = utility.read_pickle_files(
            m.record_save_path + '/val_accuracy_history.pkl')
        utility.plot_record(x=range(run_after+1, run_after+len(data)+1), y=data, xlabel="epoch", ylabel="validation accuracy",
                            title="Validation Accuracy", save_path=m.record_save_path+"/val_accuracy_history.png", show=False)

        # evaluate model
        model = m.EmotionCNN(num_classes=7)
        utility.model_validation(model, device, test_loader,
                                m.pth_save_path, m.record_save_path, file_name='0', show=False)
        utility.model_validation(model, device, test_loader,
                                m.pth_manual_save_path, m.record_save_path, file_name='1', show=False)
