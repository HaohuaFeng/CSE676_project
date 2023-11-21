import torch.nn as nn
import torch

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
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.Conv2d(out_channels, 1, kernel_size=1))
    def forward(self, x):
        a = self.seq(x)
        a = nn.functional.sigmoid(x)
        return x * a


def make_layers(block, in_channel, out_channel, num_blocks, stride):
        layers = []
        first_block = block(in_channel, out_channel, stride)
        layers.append(first_block)
        inter_channel = out_channel
        for _ in range(num_blocks-1):
            layers.append(block(inter_channel, out_channel))
        seq = nn.Sequential(*layers)
        return seq

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


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),)
        # Shortcut connection to match dimensions
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=True),
                nn.BatchNorm2d(out_channels)
            )
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        out = self.seq(x)
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class DCNN(nn.Module):
    def __init__(self, num_classes, input_channel):
        super(DCNN, self).__init__()
        self.c1=nn.Conv2d(in_channels=input_channel, out_channels=32, kernel_size=3, padding=1, stride=1)
        self.c2=nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=1)
        self.c1c2_res=nn.Conv2d(in_channels=input_channel, out_channels=64, kernel_size=3, padding=1, stride=1)
        self.mp1=nn.MaxPool2d(2)
        self.do1=nn.Dropout(0.5)
        self.c3=nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1)
        self.c4=nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1)
        self.c5=nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=1)
        self.c3c4c5_res=nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=1)
        self.mp2=nn.MaxPool2d(2)
        self.c6=nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, stride=1)
        self.c7=nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=1)
        self.c6c7_res=nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=1)
        self.mp3=nn.MaxPool2d(2)
        self.mp4=nn.MaxPool2d(2)
        self.do2=nn.Dropout(0.5)

        self.avg_pool_size = 2
        self.FC = nn.Sequential(
            nn.Linear(256*(self.avg_pool_size**2), 1024, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 2048, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(2048, 1024, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes), 
        )
        self.attention = AttentionModule(256, 256)
        self.avgpool = nn.AdaptiveAvgPool2d(self.avg_pool_size)
        self.mha = multi_head_attention(256, 8)

    def forward(self, x):
        out = self.c2(self.c1(x))
        res = self.c1c2_res(x)
        out += res
        temp = self.do1(self.mp1(out))
        out = self.c5(self.c4(self.c3(temp)))
        res = self.c3c4c5_res(temp)
        out += res
        temp = self.mp2(out)
        out = self.c7(self.c6(temp))
        res = self.c6c7_res(temp)
        out += res
        out = self.mp4(self.mp3(out))
        out = self.do2(out)
        out = self.mha(out)
        out = self.mp4(out)
        out = self.attention(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.FC(out)
        return out


def EmotionCNN(num_classes=7, input_channel=3):
    return DCNN(num_classes, input_channel)

from torchsummary import summary

model = EmotionCNN(7, 3)
summary(model, (3, 128, 128), verbose=1)
