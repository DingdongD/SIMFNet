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


class TriCNN(nn.Module):
    def __init__(self, batch_size=128, num_classes=9):
        super(TriCNN, self).__init__()
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

        self.conv3 = nn.Sequential(
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
            nn.Linear(48 * 36, 512),
            nn.Linear(512, 64),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x1 = x[0:self.batch_size, :, :, :]
        x2 = x[self.batch_size:2 * self.batch_size, :, :, :]
        x3 = x[2 * self.batch_size:3 * self.batch_size, :, :, :]

        y1 = self.conv1(x1)
        y2 = self.conv2(x2)
        y3 = self.conv3(x3)
        y = torch.cat((y1, y2, y3), dim=1)  # 128 * 48 * 6 * 6
        y = y.reshape(-1, 48 * 36)
        out = self.fc1(y)

        return out


class TriCBAM_CNN(nn.Module):
    def __init__(self, batch_size=128, num_classes=9):
        super(TriCBAM_CNN, self).__init__()
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

        self.conv3 = nn.Sequential(
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
            nn.Linear(48 * 36, 128),
            nn.Linear(128, 32),
            nn.Linear(32, num_classes)
        )

        self.cbam = CBAM(in_channel=48, reduction=8, kernel_size=3)

        self.w1 = Variable(torch.ones(1), requires_grad=True).cuda()
        self.w2 = Variable(torch.ones(1), requires_grad=True).cuda()
        self.w3 = Variable(torch.ones(1), requires_grad=True).cuda()

    def forward(self, x):
        x1 = x[0:self.batch_size, :, :, :]
        x2 = x[self.batch_size:2 * self.batch_size, :, :, :]
        x3 = x[2 * self.batch_size:3 * self.batch_size, :, :, :]

        y1 = self.conv1(x1)
        y2 = self.conv2(x2)
        y3 = self.conv3(x3)
        y = torch.cat((y1, y2, y3), dim=1)  # 128 * 48 * 6 * 6

        # y = self.w1 * y1 + self.w2 * y2 + self.w3 * y3
        yy = self.cbam(y)
        out = y + yy
        out = out.reshape(-1, 48 * 36)
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
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        out = self.decoder(x)
        return out


class Attention(nn.Module):
    def __init__(self, in_channel: int):
        super(Attention, self).__init__()
        self.w = nn.Linear(in_channel, 1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, H):
        M = self.tanh(H)  # (batch, seq_len, rnn_size)  seq_len可以理解为时间维 rnn_size为lstm输入
        alpha = self.w(M).squeeze(2)  # (batch, seq_len)
        alpha = self.softmax(alpha)  # (batch, seq_len)

        r = H * alpha.unsqueeze(2)  # (batch, seq_len, rnn_size)
        r = r.sum(dim=1)  # (batch, rnn_size)

        return r, alpha


class TriCAE1(nn.Module):
    def __init__(self, batch_size=128):
        super(TriCAE1, self).__init__()
        self.batch_size = batch_size
        self.encoder1 = Encoder()
        self.encoder2 = Encoder()
        self.encoder3 = Encoder()

        self.decoder1 = Decoder()
        self.decoder2 = Decoder()
        self.decoder3 = Decoder()

        self.lstm1 = nn.LSTM(input_size=64, hidden_size=64, num_layers=2, batch_first=True, bidirectional=False,
                             dropout=0.1)
        self.lstm2 = nn.LSTM(input_size=64, hidden_size=64, num_layers=2, batch_first=True, bidirectional=False,
                             dropout=0.1)
        self.lstm3 = nn.LSTM(input_size=64, hidden_size=64, num_layers=2, batch_first=True, bidirectional=False,
                             dropout=0.1)
        self.lstm4 = nn.LSTM(input_size=64, hidden_size=64, num_layers=2, batch_first=True, bidirectional=False,
                             dropout=0.1)

        self.conv = nn.Sequential(
            nn.Conv1d(56, 64, 4, 2, 1),  # 128 * 64 * 32
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, 2),  # 128 * 64 * 16
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, 4, 2, 1),  # 128 * 128 * 8
            nn.ReLU(inplace=True),
            nn.MaxPool1d(4, 2, 1),  # 128 * 128 * 4
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, 4, 1),
            nn.ReLU(inplace=True),
        )

        self.fc = nn.Sequential(
            nn.Linear(256, 9)
        )

    def forward(self, x):
        x1 = x[0:self.batch_size, :, :, :]
        x2 = x[self.batch_size: 2 * self.batch_size, :, :, :]
        x3 = x[2 * self.batch_size: 3 * self.batch_size, :, :, :]

        encoder_x1 = self.encoder1(x1)
        encoder_x11 = encoder_x1.transpose(3, 1)
        encoder_x12 = encoder_x11.squeeze(2)

        encoder_x2 = self.encoder2(x2)
        encoder_x21 = encoder_x2.transpose(3, 1)
        encoder_x22 = encoder_x21.squeeze(2)

        encoder_x3 = self.encoder3(x3)
        encoder_x31 = encoder_x3.transpose(3, 1)  # 128 * 56 * 64
        encoder_x32 = encoder_x31.squeeze(2)

        decoder_x1 = self.decoder1(encoder_x1)
        decoder_x2 = self.decoder2(encoder_x2)
        decoder_x3 = self.decoder3(encoder_x3)

        lstm1, hid1 = self.lstm1(encoder_x12)
        lstm2, hid2 = self.lstm2(lstm1)
        out1 = lstm1 + lstm2
        out1 = out1[:, -1, :]

        lstm3, hid3 = self.lstm3(encoder_x22)
        lstm4, hid4 = self.lstm4(lstm3)
        out2 = lstm3 + lstm4
        out2 = out2[:, -1, :]

        out3 = self.conv(encoder_x32)
        out3 = out3.squeeze(2)

        out = torch.cat((out1, out2, out3), dim=1)
        out = self.fc(out)
        return decoder_x1, decoder_x2, decoder_x3, out


class TriCAE2(nn.Module):
    def __init__(self, batch_size=128):
        super(TriCAE2, self).__init__()
        self.batch_size = batch_size
        self.encoder1 = Encoder()
        self.encoder2 = Encoder()
        self.encoder3 = Encoder()

        self.decoder1 = Decoder()
        self.decoder2 = Decoder()
        self.decoder3 = Decoder()

        self.lstm1 = nn.LSTM(input_size=64, hidden_size=64, num_layers=2, batch_first=True, bidirectional=True,
                             dropout=0.1)

        self.lstm3 = nn.LSTM(input_size=64, hidden_size=64, num_layers=2, batch_first=True, bidirectional=True,
                             dropout=0.1)

        self.attention1 = Attention(in_channel=128)
        self.attention2 = Attention(in_channel=128)

        self.conv = nn.Sequential(
            nn.Conv1d(56, 64, 4, 2, 1),  # 128 * 64 * 32
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, 2),  # 128 * 64 * 16
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, 4, 2, 1),  # 128 * 128 * 8
            nn.ReLU(inplace=True),
            nn.MaxPool1d(4, 2, 1),  # 128 * 128 * 4
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, 4, 1),
            nn.ReLU(inplace=True),
        )

        self.fc = nn.Sequential(
            nn.Linear(128 * 3, 64),
            nn.Linear(64, 9),
        )

    def forward(self, x):
        x1 = x[0:self.batch_size, :, :, :]
        x2 = x[self.batch_size: 2 * self.batch_size, :, :, :]
        x3 = x[2 * self.batch_size: 3 * self.batch_size, :, :, :]

        encoder_x1 = self.encoder1(x1)
        encoder_x11 = encoder_x1.transpose(3, 1)
        encoder_x12 = encoder_x11.squeeze(2)

        encoder_x2 = self.encoder2(x2)
        encoder_x21 = encoder_x2.transpose(3, 1)
        encoder_x22 = encoder_x21.squeeze(2)

        encoder_x3 = self.encoder3(x3)
        encoder_x31 = encoder_x3.transpose(3, 1)  # 128 * 56 * 64
        encoder_x32 = encoder_x31.squeeze(2)

        decoder_x1 = self.decoder1(encoder_x1)
        decoder_x2 = self.decoder2(encoder_x2)
        decoder_x3 = self.decoder3(encoder_x3)

        lstm1, hid1 = self.lstm1(encoder_x12)
        out1 = lstm1
        # out1 = out1[:, -1, :]
        out1, _ = self.attention1(out1)

        lstm3, hid3 = self.lstm3(encoder_x22)
        out2 = lstm3
        # out2 = out2[:, -1, :]
        out2, _ = self.attention2(out2)

        out3 = self.conv(encoder_x32)
        out3 = out3.squeeze(2)

        out = torch.cat((out1, out2, out3), dim=1)
        out = self.fc(out)
        return decoder_x1, decoder_x2, decoder_x3, out


class TriCAE3(nn.Module):
    def __init__(self, batch_size=128):
        super(TriCAE3, self).__init__()
        self.batch_size = batch_size
        self.encoder1 = Encoder()
        self.encoder2 = Encoder()
        self.encoder3 = Encoder()

        self.decoder1 = Decoder()
        self.decoder2 = Decoder()
        self.decoder3 = Decoder()

        self.lstm1 = nn.LSTM(input_size=64, hidden_size=64, num_layers=2, batch_first=True, bidirectional=True,
                             dropout=0.1)

        self.lstm3 = nn.LSTM(input_size=64, hidden_size=64, num_layers=2, batch_first=True, bidirectional=True,
                             dropout=0.1)

        self.attention1 = Attention(in_channel=128)
        self.attention2 = Attention(in_channel=128)

        self.conv = nn.Sequential(
            nn.Conv1d(56, 64, 4, 2, 1),  # 128 * 64 * 32
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, 2),  # 128 * 64 * 16
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, 4, 2, 1),  # 128 * 128 * 8
            nn.ReLU(inplace=True),
            nn.MaxPool1d(4, 2, 1),  # 128 * 128 * 4
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, 4, 1),
            nn.ReLU(inplace=True),
        )

        self.fc = nn.Sequential(
            nn.Linear(128 * 3, 64),
            nn.Linear(64, 9),
        )

    def forward(self, x):
        x1 = x[0:self.batch_size, :, :, :]
        x2 = x[self.batch_size: 2 * self.batch_size, :, :, :]
        x3 = x[2 * self.batch_size: 3 * self.batch_size, :, :, :]

        encoder_x1 = self.encoder1(x1)
        encoder_x11 = encoder_x1.transpose(3, 1)
        encoder_x12 = encoder_x11.squeeze(2)

        encoder_x2 = self.encoder2(x2)
        encoder_x21 = encoder_x2.transpose(3, 1)
        encoder_x22 = encoder_x21.squeeze(2)

        encoder_x3 = self.encoder3(x3)
        encoder_x31 = encoder_x3.transpose(3, 1)  # 128 * 56 * 1 * 64

        decoder_x1 = self.decoder1(encoder_x1)
        decoder_x2 = self.decoder2(encoder_x2)
        decoder_x3 = self.decoder3(encoder_x3)

        lstm1, hid1 = self.lstm1(encoder_x12)
        out1 = lstm1
        # out1 = out1[:, -1, :]
        out1, _ = self.attention1(out1)

        lstm3, hid3 = self.lstm3(encoder_x22)
        out2 = lstm3
        # out2 = out2[:, -1, :]
        out2, _ = self.attention2(out2)

        out3 = self.conv(encoder_x32)
        out3 = out3.squeeze(2)

        out = torch.cat((out1, out2, out3), dim=1)
        out = self.fc(out)
        return decoder_x1, decoder_x2, decoder_x3, out


class HybridNet(nn.Module):
    def __init__(self, batch_size=128, num_classes=9):
        super(HybridNet, self).__init__()
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

        self.conv3 = nn.Sequential(
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

        self.fc1 = nn.Sequential(
            nn.Linear(64, num_classes)
        )

        self.w1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.w2 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.w3 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.w1.data.fill_(0.4)
        self.w2.data.fill_(0.4)
        self.w3.data.fill_(0.2)

    def forward(self, x):
        x1 = x[0:self.batch_size, :, :, :]  # x1 x2为时间序列数据
        x2 = x[self.batch_size:2 * self.batch_size, :, :, :]
        x3 = x[2 * self.batch_size:3 * self.batch_size, :, :, :]

        y1 = 0.299 * x1[:, 0, :, :] + 0.587 * x1[:, 1, :, :] + 0.114 * x1[:, 2, :, :]  # 转灰度 128 * 224 * 224
        y1 = self.conv1(y1)  # 128 * 64 * 128
        y1 = y1.transpose(2, 1)  # 128 * 56 * 128
        out1, hidden1 = self.lstm1(y1)  # out为128*56*64
        out1 = out1[:, -1, :]

        y2 = 0.299 * x2[:, 0, :, :] + 0.587 * x2[:, 1, :, :] + 0.114 * x2[:, 2, :, :]  # 转灰度 128 * 224 * 224
        y2 = self.conv2(y2)  # 128 * 64 * 128
        y2 = y2.transpose(2, 1)  # 128 * 56 * 128
        out2, hidden2 = self.lstm1(y2)  # out为128*56*64
        out2 = out2[:, -1, :]

        y3 = self.conv3(x3)
        out3 = y3.reshape(-1, 64)
        out = self.w1 * out1 + self.w2 * out2 + self.w3 * out3

        out = self.fc1(out)
        return out



