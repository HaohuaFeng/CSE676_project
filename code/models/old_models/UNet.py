import torch.nn as nn
import torch
import torchvision.transforms.functional as F

# saving path, will change when read optimizer_name
model_name = 'UNet_'
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


class make_double_conv_layer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(make_double_conv_layer, self).__init__()
        self.dc = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.dc(x)


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.dc = make_double_conv_layer(in_channels, out_channels)

    def forward(self, x):
        x = self.maxpool(x)
        x = self.dc(x)
        return x


class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BottleneckBlock, self).__init__()
        self.dc = make_double_conv_layer(in_channels, out_channels)

    def forward(self, x):
        x = self.dc(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.convT = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.dc = make_double_conv_layer(in_channels, out_channels)

    def forward(self, x, bridge):
        x = self.convT(x)
        size = [x.shape[2], x.shape[3]]
        crop_bridge = F.center_crop(bridge, size)
        x = torch.cat([x, crop_bridge], dim=1)
        x = self.dc(x)
        return x


class EmotionCNN(nn.Module):
    def __init__(self, input_channel=1, num_classes=7):
        super(EmotionCNN, self).__init__()
        # input block
        self.input = make_double_conv_layer(input_channel, 64)
        
        # Encoder
        self.encoder2 = EncoderBlock(64, 128)
        self.encoder3 = EncoderBlock(128, 256)
        self.encoder4 = EncoderBlock(256, 512)

        # Bottleneck
        self.maxpool = nn.MaxPool2d(2)
        self.bottleneck = BottleneckBlock(512, 1024)

        # Decoder
        self.decoder1 = DecoderBlock(1024, 512)
        self.decoder2 = DecoderBlock(512, 256)
        self.decoder3 = DecoderBlock(256, 128)
        self.decoder4 = DecoderBlock(128, 64)

        # UNet output layer
        self.out = nn.Conv2d(64, num_classes, kernel_size=1)
        
        # resize the output to binary
        self.resized_output = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        input_block = self.input(x)
        e2 = self.encoder2(input_block)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        bottleneck_input = self.maxpool(e4)
        bottleneck = self.bottleneck(bottleneck_input)
        d1 = self.decoder1(bottleneck, e4)
        d2 = self.decoder2(d1, e3)
        d3 = self.decoder3(d2, e2)
        d4 = self.decoder4(d3, input_block)
        x = self.out(d4)
        x = self.resized_output(x)
        x = x.view(x.size(0), -1)
        return x

