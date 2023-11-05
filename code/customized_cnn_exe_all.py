import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle
from torch.utils.data import random_split
import models.optimizer.optimizer as optimizer
from models import Customized_cnn_ELU as m
from helper.training_early_stop import EarlyStop
import helper.utility as utility
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau
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
    optimizer_configs = ['Adam_amsgrad', 'SGD']

    for optimizer_name in optimizer_configs:

        '''
        0. Data Pre-processing
        '''

        data_transforms = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])  # normalize
        ])

        train_dataset = datasets.ImageFolder(
            '../dataset/train', transform=data_transforms)
        # split training set to training set and validation set
        # a random seed to ensure reproducibility of results.
        torch.manual_seed(42)
        train_size = int(0.9 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = random_split(
            train_dataset, [train_size, val_size])

        test_dataset = datasets.ImageFolder(
            '../dataset/test', transform=data_transforms)


        train_loader = DataLoader(train_dataset, batch_size=64,
                                shuffle=True, num_workers=16, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=64,
                                shuffle=False, num_workers=16, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=32,
                                shuffle=False, num_workers=16, pin_memory=False)

        print(len(val_loader), len(train_dataset))

        # select device
        device = utility.select_devices(use_cudnn_if_avaliable=True)

        '''
        1. Model
        '''
        # Be used to compare results.

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
        optimizer_name = "Adam_amsgrad"
        # optimizer_name = "SGD"

        # saving path
        m.model_name += m.model_name + optimizer_name
        m.pth_save_path = './model_data/' + m.model_name + '/model.pth'
        m.pth_manual_save_path = './model_data/' + \
            m.model_name + '/manual_save_model.pth'
        m.record_save_path = './model_data/' + m.model_name

        # initialize model, loss-function and optimizer
        model = m.EmotionCNN(num_classes=7)  # FER-2013 has 7 emotion class
        if not os.path.exists(m.record_save_path):
            os.makedirs(m.record_save_path)
        criterion = nn.CrossEntropyLoss()
        optimizer = optimizer.create_optimizer(model.parameters(), optimizer_name)

        # less: num_epochs, stop_counter_interval, different
        # more: stop_counter

        # training model
        num_epochs = 500

        # early stopping variables
        stop_counter = 18  # number of count to trigger early stop (patience)
        stop_counter_window = stop_counter + 5  # a range to check stop_counter
        different = 0.00008  # different between the best val loss and the most recent one
        stop_counter_interval = 10  # check for early stop for every stop_counter_interval
        counter = 0  # number of count for every trail of early stop
        is_always = True  # always check for early stop, set to true will ignore other setting except stop_counter
        is_exe = False  # is early stop running
        run_after = 0
        early_stopping = EarlyStop(
            m.pth_save_path, stop_counter, different, type="accuracy")

        # ReduceLRonPlateau (which can improve lr every epoch)
        lr_scheduler = ReduceLROnPlateau(
            optimizer,
            mode='max',                 # 'max' for monitoring validation accuracy
            factor=0.4,                 # factor by which the learning rate will be reduced
            patience=6,                 # number of epochs with no improvement to trigger LR reduction
            min_lr=1e-7,                # minimum learning rate
            verbose=1                   # (1: print messages, 0: not print message)
        )

        model.to(device)

        # progress bar
        process = tqdm(range(num_epochs),
                    bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}', colour='green', unit='epoch')

        for epoch in process:
            running_loss = 0.0
            accuracy = 0.0
            model.train()

            # Loop over the training data in batches
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                # forwarding get output
                outputs = model(inputs)

                # compute loss of output
                loss = criterion(outputs, labels)

                # backward propagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # record training status
                running_loss += loss.item()

                # calculate accuracy
                probability = torch.nn.functional.softmax(outputs, dim=1)
                max_probability, prediction = torch.max(probability, dim=1)
                num_correct_prediction = (prediction == labels).sum().item()
                accuracy += num_correct_prediction / inputs.shape[0]
                correct_prediction_pre_epoch.append(num_correct_prediction)

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

                model.eval()  # set the model to evaluation mode
                with torch.no_grad():
                    for inputs, labels in val_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        val_loss += loss.item()

                        # calculate validation accuracy
                        probability = torch.nn.functional.softmax(outputs, dim=1)
                        max_probability, prediction = torch.max(probability, dim=1)
                        num_correct_prediction = (prediction == labels).sum().item()
                        accuracy = num_correct_prediction / inputs.shape[0]
                        val_accuracy += accuracy

                val_loss = val_loss / len(val_loader)
                val_accuracy = val_accuracy / len(val_loader)
                val_loss_per_epoch.append(val_loss)
                val_accuracy_per_epoch.append(val_accuracy)

                # check the early stopping status
                early_stopping.check_status(model, val_accuracy)

                # at the end of each epoch, update the learning rate
                lr_scheduler.step(val_accuracy)

                # display recently 5 average loss of epochs
                process.set_description(f"avg loss[-5:] = {['{:.5f}'.format(num) for num in loss_history_per_epoch[-5:]]}\t"
                                        f"val loss[-5:] = {['{:.5f}'.format(num) for num in val_loss_per_epoch[-5:]]}\t"
                                        f"accuracy[-5:] = {['{:.5f}'.format(num) for num in accuracy_per_epoch[-5:]]}\t"
                                        f"val accuracy[-5:] = {['{:.5f}'.format(num) for num in val_accuracy_per_epoch[-5:]]}\t"
                                        f"best value = {'{:.5f}'.format(early_stopping.best_of_all_value)}\t"
                                        f"Counter = {early_stopping.counter}/{stop_counter} | {counter}/{stop_counter_window}\t")
            else:
                process.set_description(f"avg loss[-5:] = {['{:.5f}'.format(num) for num in loss_history_per_epoch[-5:]]}\t"
                                        f"accuracy[-5:] = {['{:.5f}'.format(num) for num in accuracy_per_epoch[-5:]]}\t")

            # Check for early stopping
            if early_stopping.early_stop:
                print('\nTrigger Early Stopping\n')
                break

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
        utility.plot_record(x=range(len(data)), y=data, xlabel="epoch", ylabel="loss", title="Training Loss",
                            save_path=m.record_save_path+"/loss_history.png")

        data = utility.read_pickle_files(m.record_save_path + '/accuracy_history.pkl')
        utility.plot_record(x=range(len(data)), y=data, xlabel="epoch", ylabel="accuracy", title="Training Accuracy",
                            save_path=m.record_save_path+"/accuracy_history.png")

        data = utility.read_pickle_files(m.record_save_path + '/val_loss_history.pkl')
        utility.plot_record(x=range(len(data)), y=data, xlabel="epoch", ylabel="validation loss",
                            title="Validation Loss", save_path=m.record_save_path+"/val_loss_history.png")

        data = utility.read_pickle_files(
            m.record_save_path + '/val_accuracy_history.pkl')
        utility.plot_record(x=range(len(data)), y=data, xlabel="epoch", ylabel="validation accuracy",
                            title="Validation Accuracy", save_path=m.record_save_path+"/val_accuracy_history.png")

        # evaluate model
        model = m.EmotionCNN(num_classes=7)
        utility.model_validation(model, device, test_loader,
                                m.pth_save_path, m.record_save_path)
        utility.model_validation(model, device, test_loader,
                                m.pth_manual_save_path, m.record_save_path)
