import torch

class EarlyStop:
    """use to determine early stop
        >>> EarlyStop(path to store model, stop count, different, type  = "loss" or "accuracy")
    """
    def __init__(self, path, loss_path, stop_count, diff, diff_loss, type = "loss"):
        self.path = path
        self.loss_path = loss_path
        self.stop_count = stop_count
        self.counter = 0
        self.different = diff
        self.loss_different = diff_loss
        self.best_of_all_value = float('inf')
        self.best_of_all_value_loss = float('inf')
        self.early_stop = False
        self.best_value = None
        self.best_value_loss = None
        self.type = type

    def save(self, model, value):
        self.best_of_all_value = value
        torch.save(model.state_dict(), self.path)

    def save_loss(self, model, value):
        self.best_of_all_value_loss = value
        torch.save(model.state_dict(), self.loss_path)

    def check_status(self, model, value, loss_value):
        if self.best_value is None and self.best_value_loss is None:
            self.counter = 0
            self.best_value = value
            self.best_value_loss = loss_value
            self.save(model, value)
            self.save_loss(model, loss_value)
            self.early_stop = False
        else:
            if  loss_value > self.best_value_loss - self.loss_different:
                if self.type == "loss":
                    self.counter += 1
                    if self.counter >= self.stop_count:
                        self.early_stop = True
                        self.counter = 0
            else:
                self.best_value_loss = loss_value
                self.save_loss(model, loss_value)
                if self.type == "loss":
                    self.counter = 0

            if  value < self.best_value + self.different:
                if self.type == "accuracy":
                    self.counter += 1
                    if self.counter >= self.stop_count:
                        self.early_stop = True
                        self.counter = 0
            else:
                self.best_value = value
                self.save(model, value)
                if self.type == "accuracy":
                    self.counter = 0
