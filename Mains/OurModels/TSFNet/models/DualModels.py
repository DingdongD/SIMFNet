import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable


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


class DualCNN(nn.Module):
    def __init__(self, num_classes=9, batch_size=128):
        super(DualCNN, self).__init__()
        self.batch_size = batch_size
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 8, 8, 4, 2),  # 128 * 8 * 56 * 56
            nn.ReLU(inplace=True),
            nn.MaxPool2d(4, 2, 1),  # 128 * 8 * 28 * 28
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 16, 4, 2, 1),  # # 128 * 16 * 14 * 14
            nn.ReLU(inplace=True),
            nn.MaxPool2d(4, 2),  # 128 * 16 * 6 * 6
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(3, 8, 8, 4, 2),  # 128 * 8 * 56 * 56
            nn.ReLU(inplace=True),
            nn.MaxPool2d(4, 2, 1),  # 128 * 8 * 28 * 28
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 16, 4, 2, 1),  # # 128 * 16 * 14 * 14
            nn.ReLU(inplace=True),
            nn.MaxPool2d(4, 2),  # 128 * 16 * 6 * 6
            nn.ReLU(inplace=True)
        )

        self.fc1 = nn.Sequential(
            nn.Linear(32 * 36, 512),
            nn.Linear(512, 64),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x1 = x[0:self.batch_size, :, :, :]
        x2 = x[self.batch_size:2 * self.batch_size, :, :, :]
        y1 = self.conv1(x1)
        y2 = self.conv2(x2)
        y = torch.cat((y1, y2), dim=1)  # 128 * 32 * 6 * 6

        y = y.reshape(-1, 32 * 36)
        out = self.fc1(y)
        return out


class DualCBAMCNN(nn.Module):
    def __init__(self, batch_size=128, num_classes=9):
        super(DualCBAMCNN, self).__init__()
        self.batch_size = batch_size
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 8, 8, 4, 2),  # 128 * 8 * 56 * 56
            nn.ReLU(inplace=True),
            nn.MaxPool2d(4, 2, 1),  # 128 * 8 * 28 * 28
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 16, 4, 2, 1),  # # 128 * 16 * 14 * 14
            nn.ReLU(inplace=True),
            nn.MaxPool2d(4, 2),  # 128 * 16 * 6 * 6
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(3, 8, 8, 4, 2),  # 128 * 8 * 56 * 56
            nn.ReLU(inplace=True),
            nn.MaxPool2d(4, 2, 1),  # 128 * 8 * 28 * 28
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 16, 4, 2, 1),  # # 128 * 16 * 14 * 14
            nn.ReLU(inplace=True),
            nn.MaxPool2d(4, 2),  # 128 * 16 * 6 * 6
            nn.ReLU(inplace=True)
        )

        self.fc1 = nn.Sequential(
            nn.Linear(16 * 36, 128),
            nn.Linear(128, 32),
            nn.Linear(32, num_classes)
        )

        self.cbam = CBAM(in_channel=16, reduction=4, kernel_size=3)
        self.w1 = Variable(torch.ones(1), requires_grad=True).cuda()
        self.w2 = Variable(torch.ones(1), requires_grad=True).cuda()

    def forward(self, x):
        x1 = x[0:self.batch_size, :, :, :]
        x2 = x[self.batch_size:2 * self.batch_size, :, :, :]
        y1 = self.conv1(x1)
        y2 = self.conv2(x2)
        y = self.w1 * y1 + self.w2 * y2  # 128 * 16 * 6 * 6
        yy = self.cbam(y)
        out = y + yy
        out = out.reshape(-1, 16 * 36)
        out = self.fc1(out)
        return out


class DualCNNLSTM(nn.Module):
    def __init__(self, batch_size=128, num_classes=9):
        super(DualCNNLSTM, self).__init__()
        self.batch_size = batch_size
        self.conv1 = nn.Sequential(
            nn.Conv1d(3, 1, 8, 4, 2),  # 128 * 1 * 56
            nn.ReLU(inplace=True),
            nn.MaxPool1d(4, 2, 1),  # 128 * 1 * 28
            nn.ReLU(inplace=True),
            nn.Conv1d(3, 1, 3, 1, 1),  # 128 * 1 * 28
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(1, 1, 8, 4, 2),  # 128 * 1 * 56
            nn.ReLU(inplace=True),
            nn.MaxPool1d(4, 2, 1),  # 128 * 1 * 28
            nn.ReLU(inplace=True),
            nn.Conv1d(1, 1, 3, 1, 1),  # 128 * 1 * 28
            nn.ReLU(inplace=True)
        )
        self.lstm1 = nn.LSTM(input_size=28, hidden_size=12, num_layers=2, batch_first=True, bidirectional=True,
                             dropout=0.2)
        self.lstm2 = nn.LSTM(input_size=28, hidden_size=12, num_layers=2, batch_first=True, bidirectional=True,
                             dropout=0.2)
        self.fc1 = nn.Sequential(
            nn.Linear(224 * 12 * 2, 1024),
            nn.Linear(1024, 192),
            nn.Linear(192, num_classes)
        )
        self.w1 = Variable(torch.ones(1), requires_grad=True).cuda()
        self.w2 = Variable(torch.ones(1), requires_grad=True).cuda()

    def forward(self, x):
        x1 = x[0:self.batch_size, :, :, :]
        x2 = x[self.batch_size:2 * self.batch_size, :, :, :]

        y1, y2 = torch.zeros(128, 1, 224, 28), torch.zeros(128, 1, 224, 28)

        for t in range(224):
            y1[:, :, t, :] = self.conv1(x1[:, :, t, :])
            y2[:, :, t, :] = self.conv2(x2[:, :, t, :])
        y1 = y1.squeeze(1)
        y2 = y2.squeeze(1)
        y1 = y1.cuda()
        y2 = y2.cuda()
        out1, hidden1 = self.lstm1(y1)
        out2, hidden2 = self.lstm2(y2)
        # out = torch.cat((out1, out2), dim=1)
        out = self.w1 * out1 + self.w2 * out2
        out = out.reshape(-1, 224 * 12 * 2)

        out = self.fc1(out)
        return out


class HybridNet1(nn.Module):  # DUALCNNLSTM为HYBRIDNET
    def __init__(self, batch_size=128):
        super(HybridNet1, self).__init__()
        self.batch_size = batch_size

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

        self.conv2 = nn.Sequential(
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

        self.lstm2 = nn.LSTM(input_size=64, hidden_size=32, num_layers=3, batch_first=True, bidirectional=True)

        self.w1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.w2 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.w1.data.fill_(0.3)
        self.w2.data.fill_(0.3)

        self.fc = nn.Sequential(
            nn.Linear(64, 9)
        )

    def forward(self, x):
        x1 = x[0:self.batch_size, :, :, :]
        x2 = x[self.batch_size:2 * self.batch_size, :, :, :]

        y1 = 0.299 * x1[:, 0, :, :] + 0.587 * x1[:, 1, :, :] + 0.114 * x1[:, 2, :, :]  # 转灰度 128 * 224 * 224
        y1 = self.conv1(y1)  # 128 * 64 * 128
        y1 = y1.transpose(2, 1)  # 128 * 56 * 128

        y2 = 0.299 * x2[:, 0, :, :] + 0.587 * x2[:, 1, :, :] + 0.114 * x2[:, 2, :, :]  # 转灰度 128 * 224 * 224
        y2 = self.conv2(y2)  # 128 * 64 * 128
        y2 = y2.transpose(2, 1)  # 128 * 56 * 128

        out1, hidden1 = self.lstm1(y1)  # out为128*56*64
        out1 = out1[:, -1, :]
        out2, hidden2 = self.lstm1(y2)  # out为128*56*64
        out2 = out2[:, -1, :]

        out = self.w1 * out1 + self.w2 * out2
        out = self.fc(out)
        return out


class HybridNet2(nn.Module):  # DUALCNNLSTM为HYBRIDNET
    def __init__(self, batch_size=128):
        super(HybridNet2, self).__init__()
        self.batch_size = batch_size

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

        self.conv2 = nn.Sequential(
            nn.Conv2d(3, 8, 4, 2, 1),  # 128 * 8 * 112 * 112
            nn.ReLU(inplace=True),
            nn.MaxPool2d(4, 2, 1),  # 128 * 8 * 56 * 56
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 16, 2, 2),  # # 128 * 16 * 28 * 28
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 128 * 16 * 14 * 14
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 4, 2, 1),  # # 128 * 16 * 7 * 7
            nn.ReLU(inplace=True),
            nn.MaxPool2d(4, 2),  # 128 * 16 * 2 * 2
            nn.ReLU(inplace=True),

        )

        self.fc = nn.Sequential(
            nn.Linear(64, 9)
        )

        self.w1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.w2 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.w1.data.fill_(0.3)
        self.w2.data.fill_(0.3)

    def forward(self, x):
        x1 = x[0:self.batch_size, :, :, :]
        x2 = x[self.batch_size:2 * self.batch_size, :, :, :]

        y1 = 0.299 * x1[:, 0, :, :] + 0.587 * x1[:, 1, :, :] + 0.114 * x1[:, 2, :, :]  # 转灰度 128 * 224 * 224
        y1 = self.conv1(y1)  # 128 * 64 * 128
        y1 = y1.transpose(2, 1)  # 128 * 56 * 128
        out1, hidden1 = self.lstm1(y1)  # out为128*56*64
        out1 = out1[:, -1, :]

        y2 = self.conv2(x2)
        out2 = y2.reshape(-1, 64)

        out = self.w1 * out1 + self.w2 * out2
        out = self.fc(out)

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
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        out = self.decoder(x)
        return out


class RDNET1(nn.Module):
    def __init__(self, batch_size=128):
        super(RDNET1, self).__init__()
        self.batch_size = batch_size
        self.encoder1 = Encoder()
        self.encoder2 = Encoder()

        self.decoder1 = Decoder()
        self.decoder2 = Decoder()

        self.lstm1 = nn.LSTM(input_size=128, hidden_size=128, num_layers=2, batch_first=True, bidirectional=False,
                             dropout=0.1)
        self.lstm2 = nn.LSTM(input_size=128, hidden_size=128, num_layers=2, batch_first=True, bidirectional=False,
                             dropout=0.1)
        self.fc = nn.Sequential(
            nn.Linear(128, 9)
        )

    def forward(self, x):
        x1 = x[0:self.batch_size, :, :, :]
        x2 = x[self.batch_size: 2 * self.batch_size, :, :, :]

        encoder_x1 = self.encoder1(x1)
        encoder_x11 = encoder_x1.transpose(3, 1)
        encoder_x12 = encoder_x11.squeeze(2)

        encoder_x2 = self.encoder2(x2)
        encoder_x21 = encoder_x2.transpose(3, 1)
        encoder_x22 = encoder_x21.squeeze(2)

        decoder_x1 = self.decoder1(encoder_x1)
        decoder_x2 = self.decoder2(encoder_x2)

        en_x = torch.cat((encoder_x12, encoder_x22), dim=2)
        lstm1, hid1 = self.lstm1(en_x)
        lstm2, hid2 = self.lstm2(lstm1)
        out = lstm2 + lstm1
        out = self.fc(out[:, -1, :])

        return decoder_x1, decoder_x2, out

