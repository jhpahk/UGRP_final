import torch
import torch.nn as nn


class Conv3x3(nn.Module):   # Bottleneck 3x3
    def __init__(self, in_channels, out_channels, mid_channels=8, stride=1, padding=1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, stride=stride, padding=padding)
        self.conv3 = nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.conv3(out)

        return out


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=8):
        super().__init__()

        self.conv1 = Conv3x3(in_channels, out_channels)
        self.conv2 = Conv3x3(out_channels, out_channels)

        self.relu = nn.ReLU()

        if in_channels != out_channels:
            self.channel_scaler = Conv3x3(in_channels, out_channels)
        else:
            self.channel_scaler = None

    def forward(self, x):
        if self.channel_scaler is not None:
            residual = self.channel_scaler(x)
        else:
            residual = x

        x = self.conv1(x)
        x = self.relu(x)

        x = self.conv2(x)
        out = x + residual

        return out


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.avgpool = nn.AvgPool2d(kernel_size=2)

        self.res1 = ResBlock(64, 256)
        self.bn2 = nn.BatchNorm2d(256)

        self.res2 = ResBlock(256, 256)
        self.bn3 = nn.BatchNorm2d(256)

        self.res3 = ResBlock(256, 32)
        
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        f2 = self.avgpool(x)     # 1/2, 64 channels

        x = self.res1(f2)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.res2(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.avgpool(x)

        x = self.res3(x)
        f4 = self.relu(x)       # 1/4, 32 channels

        return f2, f4


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_T_4_2 = nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(256)

        self.conv_T_2_1 = nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(256)

        self.skipconv_f2 = Conv3x3(64, 256)
        self.bn_f2 = nn.BatchNorm2d(256)

        self.conv_f4 = Conv3x3(32, 256)
        self.bn_f4 = nn.BatchNorm2d(256)

        self.conv_f4_2 = Conv3x3(256, 256)
        self.bn_f4_2 = nn.BatchNorm2d(256)

        self.conv_f2_1 = Conv3x3(256, 256)
        self.bn_f2_1 = nn.BatchNorm2d(256)

        self.conv_final = nn.Conv2d(256, 3, kernel_size=3, padding=1)

        self.relu = nn.ReLU()

    def forward(self, f2, f4):
        f4 = self.conv_f4(f4)
        f4 = self.bn_f4(f4)
        f4 = self.relu(f4)

        f4_2 = self.conv_T_4_2(f4)
        f4_2 = self.bn1(f4_2)
        f4_2 = self.relu(f4_2)

        f2 = self.skipconv_f2(f2)
        f2 = self.bn_f2(f2)
        f2 = self.relu(f2)

        f4_2 = f4_2 + f2
        f4_2 = self.conv_f4_2(f4_2)
        f4_2 = self.bn_f4_2(f4_2)
        f4_2 = self.relu(f4_2)

        f2_1 = self.conv_T_2_1(f4_2)
        f2_1 = self.bn2(f2_1)
        f2_1 = self.relu(f2_1)

        f2_1 = self.conv_f2_1(f2_1)
        f2_1 = self.bn_f2_1(f2_1)
        f2_1 = self.relu(f2_1)

        out = self.conv_final(f2_1)

        return out


class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        f2, f4 = self.encoder(x)
        out = self.decoder(f2, f4)

        return out