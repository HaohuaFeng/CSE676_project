import torch.nn as nn

# Reference: https://arxiv.org/ftp/arxiv/papers/2206/2206.09509.pdf
# saving path, will change when read optimizer_name
model_name = 'test_'
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
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        attention = nn.functional.sigmoid(self.conv3(x))
        return attention

class DCNN(nn.Module):
    def __init__(self, num_classes, input_channel = 1):
        super(DCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=input_channel, out_channels=512, kernel_size=5, padding=2),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 256, kernel_size=5, padding=2),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
        )

        self.avg_pool_size = 8
        self.FC = nn.Sequential(
            nn.Linear(512 * self.avg_pool_size**2, 256),
            nn.BatchNorm1d(256),
            nn.Dropout(0.25),
            nn.Linear(256, num_classes),
        )

        self.attention = AttentionModule(512, 512)
        self.avgpool = nn.AdaptiveAvgPool2d(self.avg_pool_size)

    def forward(self, x):
        x = self.features(x)
        attention = self.attention(x)
        x = attention * x
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.FC(x)
        return x

def EmotionCNN(num_classes=7, input_channel=3):
    return DCNN(num_classes, input_channel)

from torchsummary import summary

model = EmotionCNN(7, 1)
summary(model, (1, 64, 64))