import torch


class EarlyStop:
    def __init__(self, path, stop_count, diff, type = "loss"):
        self.path = path
        self.stop_count = stop_count
        self.counter = 0
        self.different = diff
        self.best_of_all_value = float('inf')
        self.early_stop = False
        self.best_value = None
        self.type = type

    def save(self, model, value):
        self.best_of_all_value = value
        torch.save(model.state_dict(), self.path)

    def check_status(self, model, value):
        if self.best_value is None:
            self.counter = 0
            self.best_value = value
            self.save(model, value)
            self.early_stop = False
        elif self.type == "loss" and (- value) < (- self.best_value + self.different):
            self.counter += 1
            if self.counter >= self.stop_count:
                self.early_stop = True
                self.best_value = None
        elif self.type == "accuracy" and value < self.best_value + self.different:
            self.counter += 1
            if self.counter >= self.stop_count:
                self.early_stop = True
                self.best_value = None
        else:
            self.best_value = value
            self.save(model, value)
            self.counter = 0
