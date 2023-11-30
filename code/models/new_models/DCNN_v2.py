import torch.nn as nn

# add attention to focus important pixel
# Reference: https://arxiv.org/ftp/arxiv/papers/2206/2206.09509.pdf
# saving path, will change when read optimizer_name
model_name = 'DCNN/v2_'
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
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, 1, kernel_size=1),
            nn.Sigmoid())
    def forward(self, x):
        a = self.seq(x)
        return x * a

class multi_head_attention(nn.Module):
    def __init__(self, channel, heads):
        super(multi_head_attention, self).__init__()
        self.mha1 = nn.MultiheadAttention(channel, heads)
    def forward(self, x):
        x = x.view(x.size(0), x.size(1), -1) # [batch, channel, seq(size*size)]
        x = x.permute(2, 0, 1) # [seq, batch, channel]
        x, _ = self.mha1(x, x, x)
        x = x.permute(1, 2, 0) # [batch, channel, seq]
        x = x.view(x.size(0), x.size(1), int(x.size(2)**0.5), -1) # [batch, channel, size, size]
        return x

class DCNN(nn.Module):
    def __init__(self, num_classes, input_channel = 1):
        super(DCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=input_channel, out_channels=32, kernel_size=3, padding=0, stride=1), 
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=0, stride=1), 
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.5),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=0, stride=1), 
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=0, stride=1), 
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=0, stride=1), 
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=0, stride=1), 
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=0, stride=1), 
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.MaxPool2d(2),
            nn.Dropout(0.5),
        )
        self.avg_pool_size = 2
        self.FC = nn.Sequential(
            nn.Linear(256*(self.avg_pool_size**2), 256, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 256, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes), 
        )
        self.attention = AttentionModule(256, 256)
        self.mhattention = multi_head_attention(channel=256, heads=8)
        self.avgpool = nn.AdaptiveAvgPool2d(self.avg_pool_size)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.features(x)
        x = self.attention(x)
        x = self.mhattention(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.FC(x)
        return x

def EmotionCNN(num_classes=7, input_channel=3):
    return DCNN(num_classes, input_channel)

from torchsummary import summary

model = EmotionCNN(7, 1)
summary(model, (1, 64, 64))