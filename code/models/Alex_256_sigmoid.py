import torch.nn as nn

# saving path, will change when read optimizer_name
model_name = 'Alex_256_sigmoid_'
pth_save_path = ''
pth_manual_save_path = ''
record_save_path = ''

def update_file_name(optimizer_name):
    global pth_save_path, pth_manual_save_path, record_save_path
    new_name = model_name + optimizer_name
    pth_save_path = './model_data/' + new_name + '/model.pth'
    pth_manual_save_path = './model_data/' + new_name + '/manual_save_model.pth'
    record_save_path = './model_data/' + new_name


class EmotionCNN(nn.Module):
    def __init__(self, num_classes):
        super(EmotionCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=96, kernel_size=11, stride=4, padding=2),
            # out_channels is decided by # of filters
            # batch_size doesn't show here and is different from in_channels.
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),  # inplace: override
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2), )
        # output shape: (batch_size, channels = 256, height = 6, width = 6)

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.flatten = nn.Flatten(1)

        # flatten from channel, ex: [batch_size, channels(1), height, width] -> [batch_size, channels * height * width]
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 256),
            nn.BatchNorm1d(256),
            nn.Sigmoid(),
            nn.Dropout(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.Sigmoid(),
            nn.Dropout(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.Sigmoid(),
            nn.Dropout(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.Sigmoid(),
            nn.Linear(256, num_classes), )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x  # the probability of 7 emotion class
