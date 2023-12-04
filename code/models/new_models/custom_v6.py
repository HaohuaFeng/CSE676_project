import torch.nn as nn

# change  FE output to 256, dropout 0.25 -> 0.5
# saving path, will change when read optimizer_name
model_name = 'custom/v6_'
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
    
        self.conv0 = nn.Conv2d(in_channels=input_channel, out_channels=512, kernel_size=5, padding=2)
        self.bn0 = nn.BatchNorm2d(512)
        self.conv0_ = nn.Conv2d(512, 256, kernel_size=5, padding=2)
        self.bn0_ = nn.BatchNorm2d(256)
        self.res0 = nn.Conv2d(in_channels=input_channel, out_channels=256, kernel_size=5, padding=2)
        self.bn_res0 = nn.BatchNorm2d(256)
        self.mp0 = nn.MaxPool2d(2)
        self.do0 = nn.Dropout(0.5)
            
        self.conv1 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv1_ = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn1_ = nn.BatchNorm2d(128)
        self.res1 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1)
        self.bn_res1 = nn.BatchNorm2d(128)
        self.mp1 = nn.MaxPool2d(2)
        self.do1 = nn.Dropout(0.5)
            
        self.conv2 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv2_ = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn2_ = nn.BatchNorm2d(512)
        self.res2 = nn.Conv2d(in_channels=128, out_channels=512, kernel_size=3, padding=1)
        self.bn_res2 = nn.BatchNorm2d(512)
        self.mp2 = nn.MaxPool2d(2)
        self.do2 = nn.Dropout(0.5)
        
        self.conv3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(512)
        self.conv3_ = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn3_ = nn.BatchNorm2d(512)
        self.res3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.bn_res3 = nn.BatchNorm2d(512)
        self.mp3 = nn.MaxPool2d(2)
        self.do3 = nn.Dropout(0.5)

        self.avg_pool_size = 4
        self.FC = nn.Sequential(
            nn.Linear(512 * self.avg_pool_size**2, 256),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),)

        self.attention = AttentionModule(512, 512)
        self.mhattention = multi_head_attention(channel=512, heads=8)
        self.resA = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.bn_resA = nn.BatchNorm2d(512)
        
        self.avgpool = nn.AdaptiveAvgPool2d(self.avg_pool_size)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.bn0(self.conv0(x))
        out = self.bn0_(self.conv0_(out))
        res = self.bn_res0(self.res0(x))
        out = self.act(out + res)
        x = self.do0(self.mp0(out))
        
        out = self.bn1(self.conv1(x))
        out = self.bn1_(self.conv1_(out))
        res = self.bn_res1(self.res1(x))
        out = self.act(out + res)
        x = self.do1(self.mp1(out))
        
        out = self.bn2(self.conv2(x))
        out = self.bn2_(self.conv2_(out))
        res = self.bn_res2(self.res2(x))
        out = self.act(out + res)
        x = self.do2(self.mp2(out))
        
        out = self.bn3(self.conv3(x))
        out = self.bn3_(self.conv3_(out))
        res = self.bn_res3(self.res3(x))
        out = self.act(out + res)
        x = self.do3(self.mp3(out))
        
        out = self.attention(x)
        out = self.mhattention(out)
        res = self.bn_resA(self.resA(x))
        out = self.act(out + res)
        
        x = self.avgpool(out)
        x = x.view(x.size(0), -1)
        x = self.FC(x)
        return x


def EmotionCNN(num_classes=7, input_channel=3):
    return DCNN(num_classes, input_channel)

from torchsummary import summary

model = EmotionCNN(7, 1)
summary(model, (1, 64, 64))
