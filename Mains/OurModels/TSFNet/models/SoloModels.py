import torch.nn as nn
import numpy as np
import torch
from sklearn import svm
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 8, 4, 2),  # 128 * 16 * 56 * 56
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 128 * 16 * 28 * 28
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 8, 4, 2, 1),  # 128 * 8 * 14 * 14
            nn.ReLU(inplace=True),
            nn.MaxPool2d(4, 2),  # 128 * 8 * 6 * 6
            nn.ReLU(inplace=True),
        )

        self.decoder = nn.Sequential(

            nn.ConvTranspose2d(8, 16, 5, 5, 1),  # 128 * 16 * 28 * 28
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 8, 2, 2),  # 128 * 8 * 56 * 56
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(8, 3, 6, 4, 1),
            nn.Tanh(),
        )

        self.autoencoder = nn.Sequential(
            self.encoder,
            self.decoder
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(36*8, 9)
        )

    def forward(self, x):
        decoder = self.autoencoder(x)
        out = self.fc(self.encoder(x))

        return decoder, out


class AlexNet(nn.Module):
    def __init__(self, num_classes=9):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),  # 128 * 64 * 56 * 56
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # 128 * 64 * 27 * 27
            nn.Conv2d(64, 192, kernel_size=5, padding=2),  # 128 * 192 * 27 * 27
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # 128 * 64 * 13 * 13
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # 128 * 256 * 6 * 6
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x


# 定义残差块ResBlock
class ResBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResBlock, self).__init__()
        # 这里定义了残差块内连续的2个卷积层
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            # shortcut，这里为了跟2个卷积层的结果结构一致，要做处理
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        # 将2个卷积层的输出跟处理过的x相加，实现ResNet的基本结构
        out = out + self.shortcut(x)
        out = F.relu(out)

        return out


class ResNet18(nn.Module):
    def __init__(self, num_classes=9):
        super(ResNet18, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.layer1 = self.make_layer(ResBlock, 64, 2, stride=1)
        self.layer2 = self.make_layer(ResBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResBlock, 512, 2, stride=2)
        self.fc = nn.Linear(512, num_classes)

    # 这个函数主要是用来，重复同一个残差块
    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        # 在这里，整个ResNet18的结构就很清晰了
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class CAM(nn.Module):
    def __init__(self, in_channel, reduction):
        super(CAM, self).__init__()
        self.in_channel = in_channel
        self.reduction = reduction
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(self.in_channel, self.in_channel // self.reduction),
            nn.ReLU(inplace=True),
            nn.Linear(self.in_channel // self.reduction, self.in_channel)
        )
        self.act = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y1 = self.avg_pool(x).view(b, c)
        y2 = self.max_pool(x).view(b, c)
        y1 = self.mlp(y1).view(b, c, 1, 1)
        y2 = self.mlp(y2).view(b, c, 1, 1)
        y = y1 + y2
        y = self.act(y)
        return x * y.expand_as(x)


class SAM(nn.Module):
    def __init__(self, kernel_size=7):
        super(SAM, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)  # max和avg后两个通道
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  # batch,1,H,W
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # batch,1,H,W
        y = torch.cat([avg_out, max_out], dim=1)  # batch,2,H,W
        y = self.conv1(y)

        return x * self.sigmoid(y).expand_as(x)


class CBAM(nn.Module):
    def __init__(self, in_channel, reduction, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = CAM(in_channel, reduction)
        self.spatial_attention = SAM(kernel_size)

    def forward(self, x):
        y = self.channel_attention(x)
        y = self.spatial_attention(y)
        return y


class SingleCNN(nn.Module):
    def __init__(self, num_classes=9):
        super(SingleCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 8, 4, 2, 1),  # 128 * 8 * 112 * 112
            nn.ReLU(inplace=True),
            nn.MaxPool2d(4, 2, 1),  # 128 * 8 * 56 * 56
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 16, 4, 2),  # 128 * 16 * 27 * 27
            nn.ReLU(inplace=True),
            nn.MaxPool2d(5, 2, 2),  # 128 * 16 * 14 * 14
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Sequential(nn.Linear(16 * 196, 128),
                                        nn.Linear(128, num_classes))

    def forward(self, x):
        y1 = self.conv1(x)
        y2 = y1.view(-1, 16 * 196)
        out = self.classifier(y2)
        return out


class CBAM_CNN(nn.Module):
    def __init__(self, num_classes=9):
        super(CBAM_CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 8, 4, 2, 1),  # 128 * 8 * 112 * 112
            nn.ReLU(inplace=True),
            nn.MaxPool2d(4, 2, 1),  # 128 * 8 * 56 * 56
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 16, 4, 2),  # 128 * 16 * 27 * 27
            nn.ReLU(inplace=True),
            nn.MaxPool2d(5, 2, 2),  # 128 * 16 * 14 * 14
            nn.ReLU(inplace=True),
        )

        self.cbam = CBAM(in_channel=16, reduction=4, kernel_size=3)

        self.fc1 = nn.Sequential(
            nn.Linear(16 * 196, 128),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        y1 = self.conv1(x)
        y2 = self.cbam(y1)
        y = F.relu(y1 + y2)
        y = y.reshape(-1, 16 * 196)
        out = self.fc1(y)

        return out


class BiLSTM(nn.Module):
    def __init__(self, out_num=112, num_classes=9):
        super(BiLSTM, self).__init__()

        self.lstm1 = nn.LSTM(input_size=224, hidden_size=out_num, num_layers=3, batch_first=True, bidirectional=True,
                             dropout=0.1)

        self.fc1 = nn.Sequential(
            nn.Linear(224 * 2, 64),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = x.squeeze(1)  # 128 * 224 * 224
        x = x.transpose(2, 1)
        out, hidden = self.lstm1(x)  # 128 * 224 * 12
        out = torch.cat((out[:, -2, :], out[:, -1, :]), dim=1)
        out = self.fc1(out)

        return out


class ResLSTM(nn.Module):
    def __init__(self, num_classes=9):
        super(ResLSTM, self).__init__()
        self.lstm1 = nn.LSTM(input_size=224, hidden_size=12, num_layers=1, batch_first=True, bidirectional=True,
                             dropout=0.2)
        self.lstm2 = nn.LSTM(input_size=24, hidden_size=12, num_layers=1, batch_first=True, bidirectional=True)

        self.fc1 = nn.Sequential(
            nn.Linear(224 * 24, 1024),
            nn.Linear(1024, 192),
            nn.Linear(192, num_classes)
        )

    def forward(self, x):
        # x = x.transpose(3, 2)
        x = x.squeeze(1)
        y1, h1 = self.lstm1(x)  # 128 * 224 * 24
        y2, h2 = self.lstm2(y1)  # 128 * 224 * 24
        y = F.relu(y1 + y2)
        # y = torch.cat((y1, y2), dim=2)
        # y = y.reshape(-1, 224 * 48)
        y = y.reshape(-1, 224 * 24)
        out = self.fc1(y)
        return out


class Conv1DLSTM(nn.Module):
    def __init__(self, num_classes=9):
        super(Conv1DLSTM, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(224, 32, 4, 2, 1),  # 128 * 32 * 112
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, 2),  # 128 * 32 * 56
            nn.Conv1d(32, 64, 3, 1, 1),  # 128 * 64 * 56
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(3, 1, 1),  # 128 * 64 * 56
            nn.ReLU(inplace=True),
        )

        self.lstm1 = nn.LSTM(input_size=64, hidden_size=32, num_layers=3, batch_first=True, bidirectional=True)

        self.fc1 = nn.Sequential(
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x = x.transpose(3, 2)
        y = x.squeeze(1)  # 沿通道维度相加 128 * 224 * 224
        y = self.conv1(y)  # 128 * 64 * 128
        y = y.transpose(2, 1)  # 128 * 56 * 128

        out, hidden = self.lstm1(y)  # out为128*112*64
        out = out[:, -1, :]
        out = self.fc1(out)
        return out


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 5, 2, 2),  # 128 * 16 * 112 * 112
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, 2, 1),  # 128 * 32 * 56 * 56
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 48, (4, 3), (2, 1), (1, 1)),  # 128 * 48 * 28 * 56
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 64, (8, 3), (4, 1), (2, 1)),  # 128 * 64 * 7 * 56
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, (7, 3), (1, 1), (0, 1)),  # 128 * 64 * 1 * 56
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        out = self.encoder(x)
        return out


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 64, (7, 3), (1, 1), (0, 1)),  # 128 * 64 * 7 * 56
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 48, (8, 3), (4, 1), (2, 1)),  # 128 * 48 * 28 * 56
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(48, 32, (4, 3), (2, 1), (1, 1)),  # 128 * 32 * 56 * 56
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, 2, 2),  # 128 * 16 * 112 * 112
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 3, 4, 2, 1),  # 128 * 16 * 224 * 224
            nn.Sigmoid(),
        )

    def forward(self, x):
        out = self.decoder(x)
        return out


class RNET1(nn.Module):
    def __init__(self, batch_size=128):
        super(RNET1, self).__init__()
        self.batch_size = batch_size
        self.encoder1 = Encoder()
        self.decoder1 = Decoder()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 56, 512),
            nn.Linear(512, 64),
            nn.Linear(64, 9),
        )

    def forward(self, x):
        encoder_x1 = self.encoder1(x)
        decoder_x1 = self.decoder1(encoder_x1)
        encoder_x2 = encoder_x1.transpose(3, 1)
        encoder_x2 = encoder_x2.squeeze(2)

        out = self.fc(encoder_x2)
        return decoder_x1, out


class RNET2(nn.Module):  # 加入LSTM
    def __init__(self):
        super(RNET2, self).__init__()
        self.encoder1 = Encoder()
        self.decoder1 = Decoder()

        self.lstm1 = nn.LSTM(input_size=64, hidden_size=64, num_layers=2, batch_first=True, bidirectional=False,
                             dropout=0.1)
        self.lstm2 = nn.LSTM(input_size=64, hidden_size=64, num_layers=2, batch_first=True, bidirectional=False,
                             dropout=0.1)
        self.fc = nn.Sequential(
            nn.Linear(64, 9)
        )

    def forward(self, x):
        encoder_x1 = self.encoder1(x)
        decoder_x1 = self.decoder1(encoder_x1)

        encoder_x2 = encoder_x1.transpose(3, 1)
        encoder_x2 = encoder_x2.squeeze(2)

        lstm1, hid1 = self.lstm1(encoder_x2)
        lstm2, hid2 = self.lstm2(lstm1)
        out = lstm2 + lstm1
        out = self.fc(out[:, -1, :])

        return decoder_x1, out
