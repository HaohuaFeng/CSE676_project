import torch.optim as optim

# use for a single run of training and evaluating all model together
def create_optimizer(parameters, optimizer_type):
    if optimizer_type == 'Adam':
        optimizer = optim.Adam(parameters, lr=0.0001)
    elif optimizer_type == 'Adam_amsgrad':
        optimizer = optim.Adam(parameters, lr=0.0001, amsgrad=True)
    elif optimizer_type == 'SGD':
        optimizer = optim.SGD(parameters, lr=0.0001, momentum=0.9)
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
    return optimizer
