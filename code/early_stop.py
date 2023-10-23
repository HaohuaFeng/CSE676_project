import torch


class EarlyStop:
    def __init__(self, filename, stop_count, diff):
        self.save_file_name = './pth_files/ES_' + filename + '.pth'
        self.stop_count = stop_count
        self.counter = 0
        self.different = diff
        self.min_val_loss = float('inf')
        self.early_stop = False
        self.best = None

    def save_model(self, validation_loss, new_model):
        torch.save(new_model.state_dict(), self.save_file_name)
        self.min_val_loss = validation_loss

    def check_status(self, validation_loss, new_model):
        if self.best is None:
            self.best = validation_loss
            self.save_model(validation_loss, new_model)
        elif (- validation_loss) < (- self.best + self.different):
            self.counter += 1
            if self.counter >= self.stop_count:
                self.early_stop = True
        else:
            self.best = validation_loss
            self.save_model(validation_loss, new_model)
            self.counter = 0
