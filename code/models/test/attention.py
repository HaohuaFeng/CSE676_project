import torch.nn as nn
import torch.nn.functional as F

# saving path, will change when read optimizer_name
model_name = 'Attention_test'
pth_save_path = ''
pth_manual_save_path = ''
record_save_path = ''
pth_save_path_loss = ''

def update_file_name(optimizer_name):
    global pth_save_path, pth_manual_save_path, record_save_path, pth_save_path_loss
    new_name = model_name + optimizer_name
    pth_save_path = './model_data/' + new_name + '/model.pth'
    pth_save_path_loss = './model_data/' + new_name + '/best_loss_model.pth'
    pth_manual_save_path = './model_data/' + new_name + '/manual_save_model.pth'
    record_save_path = './model_data/' + new_name


class AttentionModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AttentionModule, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(out_channels, 1, kernel_size=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        attention = F.sigmoid(self.conv3(x))
        return attention

class EmotionCNN(nn.Module):
    def __init__(self, num_classes, input_channel = 1):
        super(EmotionCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=input_channel, out_channels=96,
                      kernel_size=11, stride=4, padding=2),
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

        self.avgpool = nn.AdaptiveAvgPool2d((2, 2))

        # flatten from channel, ex: [batch_size, channels(1), height, width] -> [batch_size, channels * height * width]
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 2 * 2, 4096, bias=True),
            nn.BatchNorm1d(4096),
            nn.Tanh(),
            nn.Dropout(),
            nn.Linear(4096, 4096, bias=True),
            nn.BatchNorm1d(4096),
            nn.Tanh(),
            nn.Dropout(),
            nn.Linear(4096, 4096, bias=True),
            nn.BatchNorm1d(4096),
            nn.Tanh(),
            nn.Dropout(),
            nn.Linear(4096, 4096, bias=True),
            nn.BatchNorm1d(4096),
            nn.Tanh(),
            nn.Linear(4096, num_classes), )

        self.attention = AttentionModule(256, 256)

    def forward(self, x):
        features = self.features(x)
        attention = self.attention(features)
        features = features * attention
        x = self.avgpool(features)
        x = x.view(features.size(0), -1)
        x = self.classifier(x)
        return x

from torchsummary import summary

model = EmotionCNN(7, 1)
summary(model, (1, 224, 224))
