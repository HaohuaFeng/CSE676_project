import torch.nn as nn

# saving path, will change when read optimizer_name
model_name = 'Customized-cnn_ELU_'
pth_save_path = './model_data/' + model_name + '/model.pth'
pth_manual_save_path = './model_data/' + model_name + '/manual_save_model.pth'
record_save_path = './model_data/' + model_name

class EmotionCNN(nn.Module):
    def __init__(self, num_classes):
        super(EmotionCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64,
                      kernel_size=3, padding=1),
            # in_channels=1 -> gray channel
            # out_channels is decided by # of filters
            # batch_size doesn't show here and is different from in_channels.
            nn.ELU(inplace=True),  # inplace: override
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            nn.BatchNorm2d(64),

            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.3),  # get rid of 30% of data

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            nn.BatchNorm2d(128),

            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.3),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            nn.BatchNorm2d(256),

            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.3),
            # output shape: (batch_size, channels = 256, height = 6, width = 6)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(1),
            # flatten from channel, ex: [batch_size, channels(1), height, width] -> [batch_size, channels * height *
            # width]
            nn.Linear(256 * 6 * 6, 128),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
