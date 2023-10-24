import torch


class EarlyStop:
    def __init__(self, path, stop_count, diff):
        self.path = path
        self.stop_count = stop_count
        self.counter = 0
        self.different = diff
        self.min_loss = float('inf')
        self.early_stop = False
        self.best_loss = None

    def save(self, model, loss):
        self.min_loss = loss
        torch.save(model.state_dict(), self.path)

    def check_status(self, model, loss):
        if self.best_loss is None:
            self.counter = 0
            self.best_loss = loss
            self.save(model, loss)
            self.early_stop = False
        elif (- loss) < (- self.best_loss + self.different):
            self.counter += 1
            if self.counter >= self.stop_count:
                self.early_stop = True
                self.best_loss = None
        else:
            self.best_loss = loss
            self.save(model, loss)
            self.counter = 0
