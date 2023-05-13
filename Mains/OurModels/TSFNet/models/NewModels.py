import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable


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

        return r


class Encoder(nn.Module):
    def __init__(self, mode=None):
        super(Encoder, self).__init__()
        self.mode = mode

        self.encoder1 = nn.Sequential(
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

        self.encoder2 = nn.Sequential(
            nn.Conv2d(3, 8, 5, 2, 2),  # 128 * 8 * 112 * 112
            nn.ReLU(inplace=True),

            nn.Conv2d(8, 16, 4, 2, 1),  # 128 * 16 * 56 * 56
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 32, 5, 2, 2),  # 128 * 32 * 28 * 28
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 32, 4, 2, 1),  # 128 * 32 * 14 * 14
            nn.ReLU(inplace=True),

        )

    def forward(self, x):
        if self.mode == "LSTM":
            out = self.encoder1(x)
        elif self.mode == "CNN":
            out = self.encoder2(x)
        return out


class Decoder(nn.Module):
    def __init__(self, mode=None):
        super(Decoder, self).__init__()
        self.mode = mode
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, (7, 3), (1, 1), (0, 1)),  # 128 * 64 * 7 * 56
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 48, (8, 3), (4, 1), (2, 1)),  # 128 * 48 * 28 * 56
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(48, 32, (4, 3), (2, 1), (1, 1)),  # 128 * 32 * 56 * 56
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(32, 16, 2, 2),  # 128 * 16 * 112 * 112
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(16, 3, 4, 2, 1),  # 128 * 3 * 224 * 224
            nn.Sigmoid(),
        )

        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 4, 2, 1),  # 128 * 16 * 28 * 28
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 8, 6, 4, 1),  # 128 * 8 * 112 * 112
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(8, 3, 4, 2, 1),  # 128 * 3 * 224 * 224
            nn.Sigmoid(),

        )

    def forward(self, x):
        if self.mode == "LSTM":
            out = self.decoder1(x)
        elif self.mode == "CNN":
            out = self.decoder2(x)
        return out


class NewModel1(nn.Module):
    def __init__(self, batch_size=128):
        super(NewModel1, self).__init__()
        self.batch_size = batch_size
        self.encoder1 = Encoder(mode="LSTM")
        self.encoder2 = Encoder(mode="LSTM")
        self.encoder3 = Encoder(mode="CNN")

        self.decoder1 = Decoder(mode="LSTM")
        self.decoder2 = Decoder(mode="LSTM")
        self.decoder3 = Decoder(mode="CNN")

        self.lstm1 = nn.LSTM(input_size=64, hidden_size=64, num_layers=2, batch_first=True, bidirectional=False,
                             dropout=0.1)
        self.lstm2 = nn.LSTM(input_size=64, hidden_size=64, num_layers=2, batch_first=True, bidirectional=False,
                             dropout=0.1)

        self.conv = nn.Sequential(
            nn.Conv2d(32, 32, 4, 2),  # 128 * 32 * 6 * 6
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2, 2),  # 128 * 32 * 3 * 3
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, 3, 1),  # 128 * 64 * 1 * 1
            nn.ReLU(inplace=True),

            nn.Flatten(),
        )

        self.fc = nn.Sequential(
            nn.Linear(64, 9)
        )

        self.w1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.w2 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.w3 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.w1.data.fill_(1.2)
        self.w2.data.fill_(1.2)
        self.w3.data.fill_(0.6)

    def forward(self, x):
        x1 = x[0:self.batch_size, :, :, :]
        x2 = x[self.batch_size: 2 * self.batch_size, :, :, :]
        x3 = x[2 * self.batch_size: 3 * self.batch_size, :, :, :]

        encoder_x1 = self.encoder1(x1)
        encoder_x11 = encoder_x1.transpose(3, 1)
        encoder_x12 = encoder_x11.squeeze(2)  # 128 * 56 * 64

        encoder_x2 = self.encoder2(x2)
        encoder_x21 = encoder_x2.transpose(3, 1)
        encoder_x22 = encoder_x21.squeeze(2)  # 128 * 56 * 64

        encoder_x3 = self.encoder3(x3)  # 128 * 32 * 14 * 14

        decoder_x1 = self.decoder1(encoder_x1)
        decoder_x2 = self.decoder2(encoder_x2)
        decoder_x3 = self.decoder3(encoder_x3)

        lstm1, hid1 = self.lstm1(encoder_x12)  # 128 * 56 * 64
        out1 = lstm1[:, -1, :]
        lstm2, hid2 = self.lstm2(encoder_x22)  # 128 * 56 * 64
        out2 = lstm2[:, -1, :]
        out3 = self.conv(encoder_x3)  # 128 * 64

        out = self.w1 * out1 + self.w2 * out2 + self.w3 * out3

        # out = torch.cat((out1, out2), dim=1)
        out = self.fc(out)
        return decoder_x1, decoder_x2, decoder_x3, out


class NewModel2(nn.Module):
    def __init__(self, batch_size=128):
        super(NewModel2, self).__init__()
        self.batch_size = batch_size
        self.encoder1 = Encoder(mode="LSTM")
        self.encoder2 = Encoder(mode="LSTM")
        self.encoder3 = Encoder(mode="CNN")

        self.decoder1 = Decoder(mode="LSTM")
        self.decoder2 = Decoder(mode="LSTM")
        self.decoder3 = Decoder(mode="CNN")

        self.lstm1 = nn.LSTM(input_size=128, hidden_size=128, num_layers=2, batch_first=True, bidirectional=False)
        self.lstm2 = nn.LSTM(input_size=128, hidden_size=128, num_layers=2, batch_first=True, bidirectional=False)

        self.conv = nn.Sequential(
            nn.Conv2d(32, 16, 4, 2),  # 128 * 32 * 6 * 6
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2, 2),  # 128 * 32 * 3 * 3
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 32, 2, 1),  # 128 * 32 * 2 * 2
            nn.ReLU(inplace=True),

            nn.Flatten(),
        )

        self.fc = nn.Sequential(
            nn.Linear(64 * 4, 9)
        )

        self.w1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.w2 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.w3 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.w1.data.fill_(1.0)
        self.w2.data.fill_(1.0)
        self.w3.data.fill_(1.0)

    def forward(self, x):
        x1 = x[0:self.batch_size, :, :, :]
        x2 = x[self.batch_size: 2 * self.batch_size, :, :, :]
        x3 = x[2 * self.batch_size: 3 * self.batch_size, :, :, :]

        encoder_x1 = self.encoder1(x1)
        encoder_x11 = encoder_x1.transpose(3, 1)
        encoder_x12 = encoder_x11.squeeze(2)  # 128 * 56 * 64

        encoder_x2 = self.encoder2(x2)
        encoder_x21 = encoder_x2.transpose(3, 1)
        encoder_x22 = encoder_x21.squeeze(2)  # 128 * 56 * 64

        encoder_x3 = self.encoder3(x3)  # 128 * 32 * 14 * 14

        decoder_x1 = self.decoder1(encoder_x1)
        decoder_x2 = self.decoder2(encoder_x2)
        decoder_x3 = self.decoder3(encoder_x3)

        en_x = torch.cat((encoder_x12, encoder_x22), dim=2)  # 128 * 128 * 56

        lstm1, hid1 = self.lstm1(en_x)  # 128 * 56 * 64
        lstm2, hid2 = self.lstm2(lstm1)  # 128 * 56 * 64
        out2 = lstm1 + lstm2
        out2 = out2[:, -1, :]
        out3 = self.conv(encoder_x3)  # 128 * 128

        out = torch.cat((out2, out3), dim=1)

        # out = torch.cat((out1, out2), dim=1)
        out = self.fc(out)
        return decoder_x1, decoder_x2, decoder_x3, out


class NewModel3(nn.Module):
    def __init__(self, batch_size=128):
        super(NewModel3, self).__init__()
        self.batch_size = batch_size
        self.encoder1 = Encoder(mode="LSTM")
        self.encoder2 = Encoder(mode="LSTM")
        self.encoder3 = Encoder(mode="CNN")

        self.decoder1 = Decoder(mode="LSTM")
        self.decoder2 = Decoder(mode="LSTM")
        self.decoder3 = Decoder(mode="CNN")

        self.lstm1 = nn.LSTM(input_size=64, hidden_size=64, num_layers=2, batch_first=True, bidirectional=False)
        self.lstm2 = nn.LSTM(input_size=64, hidden_size=64, num_layers=2, batch_first=True, bidirectional=False)

        self.conv = nn.Sequential(
            nn.Conv2d(32, 16, 4, 2),  # 128 * 32 * 6 * 6
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2, 2),  # 128 * 32 * 3 * 3
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 32, 2, 1),  # 128 * 32 * 2 * 2
            nn.ReLU(inplace=True),

            nn.Flatten(),
        )

        self.fc = nn.Sequential(
            nn.Linear(64 * 4, 9)
        )

        self.w1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.w2 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.w3 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.w1.data.fill_(1.0)
        self.w2.data.fill_(1.0)
        self.w3.data.fill_(1.0)

    def forward(self, x):
        x1 = x[0:self.batch_size, :, :, :]
        x2 = x[self.batch_size: 2 * self.batch_size, :, :, :]
        x3 = x[2 * self.batch_size: 3 * self.batch_size, :, :, :]

        encoder_x1 = self.encoder1(x1)
        encoder_x11 = encoder_x1.transpose(3, 1)
        encoder_x12 = encoder_x11.squeeze(2)  # 128 * 56 * 64

        encoder_x2 = self.encoder2(x2)
        encoder_x21 = encoder_x2.transpose(3, 1)
        encoder_x22 = encoder_x21.squeeze(2)  # 128 * 56 * 64

        encoder_x3 = self.encoder3(x3)  # 128 * 32 * 14 * 14

        decoder_x1 = self.decoder1(encoder_x1)
        decoder_x2 = self.decoder2(encoder_x2)
        decoder_x3 = self.decoder3(encoder_x3)

        lstm1, hid1 = self.lstm1(encoder_x12)  # 128 * 56 * 64
        out1 = lstm1[:, -1, :]
        lstm2, hid2 = self.lstm2(encoder_x22)  # 128 * 56 * 64
        out2 = lstm2[:, -1, :]
        out3 = self.conv(encoder_x3)  # 128 * 128

        out = torch.cat((out1, out2, out3), dim=1)

        # out = torch.cat((out1, out2), dim=1)
        out = self.fc(out)
        return decoder_x1, decoder_x2, decoder_x3, out


class NewModel4(nn.Module):
    def __init__(self, batch_size=128):
        super(NewModel4, self).__init__()
        self.batch_size = batch_size
        self.encoder1 = Encoder(mode="LSTM")
        self.encoder2 = Encoder(mode="LSTM")
        self.encoder3 = Encoder(mode="CNN")

        self.decoder1 = Decoder(mode="LSTM")
        self.decoder2 = Decoder(mode="LSTM")
        self.decoder3 = Decoder(mode="CNN")

        self.lstm1 = nn.LSTM(input_size=64, hidden_size=64, num_layers=2, batch_first=True, bidirectional=False)
        self.lstm2 = nn.LSTM(input_size=64, hidden_size=64, num_layers=2, batch_first=True, bidirectional=False)

        self.conv = nn.Sequential(
            nn.Conv2d(32, 32, 4, 2),  # 128 * 32 * 6 * 6
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2, 2),  # 128 * 32 * 3 * 3
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, 2, 1),  # 128 * 64 * 1 * 1
            nn.ReLU(inplace=True),

            nn.Flatten(),
        )

        self.fc = nn.Sequential(
            nn.Linear(64, 9)
        )

        self.w1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.w2 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.w3 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.w1.data.fill_(1.0)
        self.w2.data.fill_(1.0)
        self.w3.data.fill_(1.0)

    def forward(self, x):
        x1 = x[0:self.batch_size, :, :, :]
        x2 = x[self.batch_size: 2 * self.batch_size, :, :, :]
        x3 = x[2 * self.batch_size: 3 * self.batch_size, :, :, :]

        encoder_x1 = self.encoder1(x1)
        encoder_x11 = encoder_x1.transpose(3, 1)
        encoder_x12 = encoder_x11.squeeze(2)  # 128 * 56 * 64

        encoder_x2 = self.encoder2(x2)
        encoder_x21 = encoder_x2.transpose(3, 1)
        encoder_x22 = encoder_x21.squeeze(2)  # 128 * 56 * 64

        encoder_x3 = self.encoder3(x3)  # 128 * 32 * 14 * 14

        decoder_x1 = self.decoder1(encoder_x1)
        decoder_x2 = self.decoder2(encoder_x2)
        decoder_x3 = self.decoder3(encoder_x3)

        lstm1, hid1 = self.lstm1(encoder_x12)  # 128 * 56 * 64
        out1 = lstm1[:, -1, :]
        lstm2, hid2 = self.lstm2(encoder_x22)  # 128 * 56 * 64
        out2 = lstm2[:, -1, :]
        out3 = self.conv(encoder_x3)  # 128 * 64

        out = self.w1 * out1 + self.w2 * out2 + self.w3 * out3

        # out = torch.cat((out1, out2), dim=1)
        out = self.fc(out)
        return decoder_x1, decoder_x2, decoder_x3, out


class NewModel5(nn.Module):
    def __init__(self, batch_size=128):
        super(NewModel5, self).__init__()
        self.batch_size = batch_size
        self.encoder1 = Encoder(mode="LSTM")
        self.encoder2 = Encoder(mode="LSTM")
        self.encoder3 = Encoder(mode="CNN")

        self.decoder1 = Decoder(mode="LSTM")
        self.decoder2 = Decoder(mode="LSTM")
        self.decoder3 = Decoder(mode="CNN")

        self.lstm1 = nn.LSTM(input_size=64, hidden_size=64, num_layers=2, batch_first=True, bidirectional=False)
        self.lstm2 = nn.LSTM(input_size=64, hidden_size=64, num_layers=2, batch_first=True, bidirectional=False)

        self.conv = nn.Sequential(
            nn.Conv2d(32, 32, 4, 2),  # 128 * 32 * 6 * 6
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2, 2),  # 128 * 32 * 3 * 3
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, 3, 1),  # 128 * 64 * 1 * 1
            nn.ReLU(inplace=True),

            nn.Flatten(),
        )

        self.fc = nn.Sequential(
            nn.Linear(64, 9)
        )

        self.attention1 = Attention(in_channel=64)
        self.attention2 = Attention(in_channel=64)

        self.w1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.w2 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.w3 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.w1.data.fill_(1.0)
        self.w2.data.fill_(1.0)
        self.w3.data.fill_(1.0)

    def forward(self, x):
        x1 = x[0:self.batch_size, :, :, :]
        x2 = x[self.batch_size: 2 * self.batch_size, :, :, :]
        x3 = x[2 * self.batch_size: 3 * self.batch_size, :, :, :]

        encoder_x1 = self.encoder1(x1)
        encoder_x11 = encoder_x1.transpose(3, 1)
        encoder_x12 = encoder_x11.squeeze(2)  # 128 * 56 * 64

        encoder_x2 = self.encoder2(x2)
        encoder_x21 = encoder_x2.transpose(3, 1)
        encoder_x22 = encoder_x21.squeeze(2)  # 128 * 56 * 64

        encoder_x3 = self.encoder3(x3)  # 128 * 32 * 14 * 14

        decoder_x1 = self.decoder1(encoder_x1)
        decoder_x2 = self.decoder2(encoder_x2)
        decoder_x3 = self.decoder3(encoder_x3)

        lstm1, hid1 = self.lstm1(encoder_x12)  # 128 * 56 * 64
        out1 = self.attention1(lstm1)
        lstm2, hid2 = self.lstm2(encoder_x22)  # 128 * 56 * 64
        out2 = self.attention2(lstm2)
        out3 = self.conv(encoder_x3)  # 128 * 64

        out = self.w1 * out1 + self.w2 * out2 + self.w3 * out3

        # out = torch.cat((out1, out2), dim=1)
        out = self.fc(out)
        return decoder_x1, decoder_x2, decoder_x3, out


class NewModel6(nn.Module):
    def __init__(self, batch_size=128):
        super(NewModel6, self).__init__()
        self.batch_size = batch_size
        self.encoder1 = Encoder(mode="LSTM")
        self.encoder2 = Encoder(mode="LSTM")
        self.encoder3 = Encoder(mode="CNN")

        self.decoder1 = Decoder(mode="LSTM")
        self.decoder2 = Decoder(mode="LSTM")
        self.decoder3 = Decoder(mode="CNN")

        self.lstm1 = nn.LSTM(input_size=128, hidden_size=128, num_layers=2, batch_first=True, bidirectional=False)
        self.lstm2 = nn.LSTM(input_size=128, hidden_size=128, num_layers=2, batch_first=True, bidirectional=False)

        self.conv = nn.Sequential(
            nn.Conv2d(32, 32, 4, 2),  # 128 * 32 * 6 * 6
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2, 2),  # 128 * 32 * 3 * 3
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, 3, 1),  # 128 * 64 * 1 * 1
            nn.ReLU(inplace=True),

            nn.Flatten(),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(3648, 512),
            nn.Linear(512, 64)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(192, 9)
        )

        self.attention1 = Attention(in_channel=64)
        self.attention2 = Attention(in_channel=64)

        self.w1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.w2 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.w3 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.w1.data.fill_(1.0)
        self.w2.data.fill_(1.0)
        self.w3.data.fill_(1.0)

    def forward(self, x):
        x1 = x[0:self.batch_size, :, :, :]
        x2 = x[self.batch_size: 2 * self.batch_size, :, :, :]
        x3 = x[2 * self.batch_size: 3 * self.batch_size, :, :, :]

        encoder_x1 = self.encoder1(x1)
        encoder_x11 = encoder_x1.transpose(3, 1)
        encoder_x12 = encoder_x11.squeeze(2)  # 128 * 56 * 64

        encoder_x2 = self.encoder2(x2)
        encoder_x21 = encoder_x2.transpose(3, 1)
        encoder_x22 = encoder_x21.squeeze(2)  # 128 * 56 * 64

        encoder_x3 = self.encoder3(x3)  # 128 * 32 * 14 * 14

        decoder_x1 = self.decoder1(encoder_x1)
        decoder_x2 = self.decoder2(encoder_x2)
        decoder_x3 = self.decoder3(encoder_x3)

        encoder_x31 = self.conv(encoder_x3)  # 128 * 64
        encoder_x32 = encoder_x31.unsqueeze(1)

        out1 = torch.cat((encoder_x12, encoder_x22), dim=2)  # 时间上的信息拼接 128 * 56 * 128
        out2 = torch.cat((encoder_x12, encoder_x32), dim=1)  # 时空拼接 128 * 57 * 64

        lstm1, hid1 = self.lstm1(out1)  # 128 * 56 * 64
        lstm2, hid2 = self.lstm1(lstm1)  # 128 * 56 * 64
        out_12 = lstm1 + lstm2
        out_12 = out_12[:, -1, :]

        out_13 = out2.reshape(-1, 3648)
        out_13 = self.fc1(out_13)  # 128 * 64

        out = torch.cat((out_12, out_13), dim=1)

        # out = torch.cat((out1, out2), dim=1)
        out = self.fc2(out)
        return decoder_x1, decoder_x2, decoder_x3, out


class NewModel7(nn.Module):
    def __init__(self, batch_size=128):
        super(NewModel7, self).__init__()
        self.batch_size = batch_size
        self.encoder1 = Encoder(mode="LSTM")
        self.encoder2 = Encoder(mode="LSTM")
        self.encoder3 = Encoder(mode="CNN")

        self.decoder1 = Decoder(mode="LSTM")
        self.decoder2 = Decoder(mode="LSTM")
        self.decoder3 = Decoder(mode="CNN")

        self.lstm1 = nn.LSTM(input_size=64, hidden_size=64, num_layers=2, batch_first=True, bidirectional=False)
        self.lstm2 = nn.LSTM(input_size=64, hidden_size=64, num_layers=2, batch_first=True, bidirectional=False)

        self.conv = nn.Sequential(
            nn.Conv2d(32, 32, 4, 2),  # 128 * 32 * 6 * 6
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2, 2),  # 128 * 32 * 3 * 3
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, 3, 1),  # 128 * 64 * 1 * 1
            nn.ReLU(inplace=True),

            nn.Flatten(),
        )

        self.fc = nn.Sequential(
            nn.Linear(64 * 3, 9)
        )

        self.attention1 = Attention(in_channel=64)
        self.attention2 = Attention(in_channel=64)

        self.w1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.w2 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.w3 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.w1.data.fill_(1.0)
        self.w2.data.fill_(1.0)
        self.w3.data.fill_(1.0)

    def forward(self, x):
        x1 = x[0:self.batch_size, :, :, :]
        x2 = x[self.batch_size: 2 * self.batch_size, :, :, :]
        x3 = x[2 * self.batch_size: 3 * self.batch_size, :, :, :]

        encoder_x1 = self.encoder1(x1)
        encoder_x11 = encoder_x1.transpose(3, 1)
        encoder_x12 = encoder_x11.squeeze(2)  # 128 * 56 * 64

        encoder_x2 = self.encoder2(x2)
        encoder_x21 = encoder_x2.transpose(3, 1)
        encoder_x22 = encoder_x21.squeeze(2)  # 128 * 56 * 64

        encoder_x3 = self.encoder3(x3)  # 128 * 32 * 14 * 14

        decoder_x1 = self.decoder1(encoder_x1)
        decoder_x2 = self.decoder2(encoder_x2)
        decoder_x3 = self.decoder3(encoder_x3)

        lstm1, hid1 = self.lstm1(encoder_x12)  # 128 * 56 * 64
        out1 = self.attention1(lstm1)
        lstm2, hid2 = self.lstm2(encoder_x22)  # 128 * 56 * 64
        out2 = self.attention2(lstm2)

        out3 = self.conv(encoder_x3)  # 128 * 64

        # out = self.w1 * out1 + self.w2 * out2 + self.w3 * out3

        out = torch.cat((out1, out2, out3), dim=1)
        out = self.fc(out)
        return decoder_x1, decoder_x2, decoder_x3, out


class NewModel8(nn.Module):
    def __init__(self, batch_size=128):
        super(NewModel8, self).__init__()
        self.batch_size = batch_size
        self.encoder1 = Encoder(mode="LSTM")
        self.encoder2 = Encoder(mode="LSTM")
        self.encoder3 = Encoder(mode="CNN")

        self.decoder1 = Decoder(mode="LSTM")
        self.decoder2 = Decoder(mode="LSTM")
        self.decoder3 = Decoder(mode="CNN")

        self.lstm1 = nn.LSTM(input_size=128, hidden_size=128, num_layers=2, batch_first=True, bidirectional=False)
        self.lstm2 = nn.LSTM(input_size=128, hidden_size=128, num_layers=2, batch_first=True, bidirectional=False)

        self.conv = nn.Sequential(
            nn.Conv2d(32, 32, 4, 2),  # 128 * 32 * 6 * 6
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2, 2),  # 128 * 32 * 3 * 3
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, 3, 1),  # 128 * 64 * 1 * 1
            nn.ReLU(inplace=True),

            nn.Flatten(),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(3648, 512),
            nn.Linear(512, 64)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(192, 9)
        )

        self.attention1 = Attention(in_channel=64)
        self.attention2 = Attention(in_channel=64)

        self.w1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.w2 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.w3 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.w1.data.fill_(1.0)
        self.w2.data.fill_(1.0)
        self.w3.data.fill_(1.0)

    def forward(self, x):
        x1 = x[0:self.batch_size, :, :, :]
        x2 = x[self.batch_size: 2 * self.batch_size, :, :, :]
        x3 = x[2 * self.batch_size: 3 * self.batch_size, :, :, :]

        encoder_x1 = self.encoder1(x1)
        encoder_x11 = encoder_x1.transpose(3, 1)
        encoder_x12 = encoder_x11.squeeze(2)  # 128 * 56 * 64

        encoder_x2 = self.encoder2(x2)
        encoder_x21 = encoder_x2.transpose(3, 1)
        encoder_x22 = encoder_x21.squeeze(2)  # 128 * 56 * 64

        encoder_x3 = self.encoder3(x3)  # 128 * 32 * 14 * 14

        decoder_x1 = self.decoder1(encoder_x1)
        decoder_x2 = self.decoder2(encoder_x2)
        decoder_x3 = self.decoder3(encoder_x3)

        encoder_x31 = self.conv(encoder_x3)  # 128 * 64
        encoder_x32 = encoder_x31.unsqueeze(1)

        out1 = torch.cat((encoder_x12, encoder_x22), dim=2)  # 时间上的信息拼接 128 * 56 * 128
        out2 = torch.cat((encoder_x22, encoder_x32), dim=1)  # 时空拼接 128 * 57 * 64

        lstm1, hid1 = self.lstm1(out1)  # 128 * 56 * 64
        lstm2, hid2 = self.lstm1(lstm1)  # 128 * 56 * 64
        out_12 = lstm1 + lstm2
        out_12 = out_12[:, -1, :]

        out_13 = out2.reshape(-1, 3648)
        out_13 = self.fc1(out_13)  # 128 * 64

        out = torch.cat((out_12, out_13), dim=1)

        # out = torch.cat((out1, out2), dim=1)
        out = self.fc2(out)
        return decoder_x1, decoder_x2, decoder_x3, out


class NewModel9(nn.Module):
    def __init__(self, batch_size=128):
        super(NewModel9, self).__init__()
        self.batch_size = batch_size
        self.encoder1 = Encoder(mode="LSTM")
        self.encoder2 = Encoder(mode="LSTM")
        self.encoder3 = Encoder(mode="CNN")

        self.decoder1 = Decoder(mode="LSTM")
        self.decoder2 = Decoder(mode="LSTM")
        self.decoder3 = Decoder(mode="CNN")

        self.lstm1 = nn.LSTM(input_size=128, hidden_size=128, num_layers=2, batch_first=True, bidirectional=False)
        self.lstm2 = nn.LSTM(input_size=128, hidden_size=128, num_layers=2, batch_first=True, bidirectional=False)

        self.conv = nn.Sequential(
            nn.Conv2d(32, 32, 4, 2),  # 128 * 32 * 6 * 6
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2, 2),  # 128 * 32 * 3 * 3
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, 3, 1),  # 128 * 64 * 1 * 1
            nn.ReLU(inplace=True),

            nn.Flatten(),
        )

        self.fc1 = nn.Sequential(
            nn.Conv1d(57, 16, 4, 2, 1),  # 128 * 16 * 32
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(16, 32, 4, 2, 1),  # 128 * 32 * 8
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(4, 2, 1),  # 128 * 32 * 4
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, 4, 1),
            nn.ReLU(inplace=True),
            nn.Flatten()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(192, 9)
        )

        self.attention1 = Attention(in_channel=64)
        self.attention2 = Attention(in_channel=64)

        self.w1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.w2 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.w3 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.w1.data.fill_(1.0)
        self.w2.data.fill_(1.0)
        self.w3.data.fill_(1.0)

    def forward(self, x):
        x1 = x[0:self.batch_size, :, :, :]
        x2 = x[self.batch_size: 2 * self.batch_size, :, :, :]
        x3 = x[2 * self.batch_size: 3 * self.batch_size, :, :, :]

        encoder_x1 = self.encoder1(x1)
        encoder_x11 = encoder_x1.transpose(3, 1)
        encoder_x12 = encoder_x11.squeeze(2)  # 128 * 56 * 64

        encoder_x2 = self.encoder2(x2)
        encoder_x21 = encoder_x2.transpose(3, 1)
        encoder_x22 = encoder_x21.squeeze(2)  # 128 * 56 * 64

        encoder_x3 = self.encoder3(x3)  # 128 * 32 * 14 * 14

        decoder_x1 = self.decoder1(encoder_x1)
        decoder_x2 = self.decoder2(encoder_x2)
        decoder_x3 = self.decoder3(encoder_x3)

        encoder_x31 = self.conv(encoder_x3)  # 128 * 64
        encoder_x32 = encoder_x31.unsqueeze(1)

        out1 = torch.cat((encoder_x12, encoder_x22), dim=2)  # 时间上的信息拼接 128 * 56 * 128
        out2 = torch.cat((encoder_x12, encoder_x32), dim=1)  # 时空拼接 128 * 57 * 64

        lstm1, hid1 = self.lstm1(out1)  # 128 * 56 * 64
        lstm2, hid2 = self.lstm1(lstm1)  # 128 * 56 * 64
        out_12 = lstm1 + lstm2
        out_12 = out_12[:, -1, :]

        out_13 = self.fc1(out2)  # 128 * 64

        out = torch.cat((out_12, out_13), dim=1)

        # out = torch.cat((out1, out2), dim=1)
        out = self.fc2(out)
        return decoder_x1, decoder_x2, decoder_x3, out


class NewModel10(nn.Module):
    def __init__(self, batch_size=128):
        super(NewModel10, self).__init__()
        self.batch_size = batch_size
        self.encoder1 = Encoder(mode="LSTM")
        self.encoder2 = Encoder(mode="LSTM")
        self.encoder3 = Encoder(mode="CNN")

        self.decoder1 = Decoder(mode="LSTM")
        self.decoder2 = Decoder(mode="LSTM")
        self.decoder3 = Decoder(mode="CNN")

        self.lstm1 = nn.GRU(input_size=128, hidden_size=64, num_layers=2, batch_first=True, bidirectional=False)
        self.lstm2 = nn.GRU(input_size=128, hidden_size=64, num_layers=2, batch_first=True, bidirectional=False)

        self.conv = nn.Sequential(
            nn.Conv2d(32, 32, 4, 2),  # 128 * 32 * 6 * 6
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2, 2),  # 128 * 32 * 3 * 3
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, 3, 1),  # 128 * 64 * 1 * 1
            nn.ReLU(inplace=True),

            nn.Flatten(),
        )

        self.fc1 = nn.Sequential(
            nn.Conv1d(57, 16, 4, 2, 1),  # 128 * 16 * 32
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(16, 32, 4, 2, 1),  # 128 * 32 * 8
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(4, 2, 1),  # 128 * 32 * 4
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, 4, 1),
            nn.ReLU(inplace=True),
            nn.Flatten()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(128, 9)
        )

        self.attention1 = Attention(in_channel=64)

        self.w1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.w2 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.w3 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.w1.data.fill_(1.0)
        self.w2.data.fill_(1.0)
        self.w3.data.fill_(1.0)

    def forward(self, x):
        x1 = x[0:self.batch_size, :, :, :]
        x2 = x[self.batch_size: 2 * self.batch_size, :, :, :]
        x3 = x[2 * self.batch_size: 3 * self.batch_size, :, :, :]

        encoder_x1 = self.encoder1(x1)
        encoder_x11 = encoder_x1.transpose(3, 1)
        encoder_x12 = encoder_x11.squeeze(2)  # 128 * 56 * 64

        encoder_x2 = self.encoder2(x2)
        encoder_x21 = encoder_x2.transpose(3, 1)
        encoder_x22 = encoder_x21.squeeze(2)  # 128 * 56 * 64

        encoder_x3 = self.encoder3(x3)  # 128 * 32 * 14 * 14

        decoder_x1 = self.decoder1(encoder_x1)
        decoder_x2 = self.decoder2(encoder_x2)
        decoder_x3 = self.decoder3(encoder_x3)

        encoder_x31 = self.conv(encoder_x3)  # 128 * 64
        encoder_x32 = encoder_x31.unsqueeze(1)

        out1 = torch.cat((encoder_x12, encoder_x22), dim=2)  # 时间上的信息拼接 128 * 56 * 128
        out2 = torch.cat((encoder_x12, encoder_x32), dim=1)  # 时空拼接 128 * 57 * 64

        lstm1, hid1 = self.lstm1(out1)  # 128 * 56 * 64
        out_12 = self.attention1(lstm1)  # 128 * 64

        out_13 = self.fc1(out2)  # 128 * 64

        out = torch.cat((out_12, out_13), dim=1)

        # out = torch.cat((out1, out2), dim=1)
        out = self.fc2(out)
        return decoder_x1, decoder_x2, decoder_x3, out


class NewModel11(nn.Module):
    def __init__(self, batch_size=128):
        super(NewModel11, self).__init__()
        self.batch_size = batch_size
        self.encoder1 = Encoder(mode="LSTM")
        self.encoder2 = Encoder(mode="LSTM")
        self.encoder3 = Encoder(mode="CNN")

        self.decoder1 = Decoder(mode="LSTM")
        self.decoder2 = Decoder(mode="LSTM")
        self.decoder3 = Decoder(mode="CNN")

        self.lstm1 = nn.GRU(input_size=128, hidden_size=64, num_layers=2, batch_first=True, bidirectional=False)
        self.lstm2 = nn.GRU(input_size=128, hidden_size=64, num_layers=2, batch_first=True, bidirectional=False)

        self.conv = nn.Sequential(
            nn.Conv2d(32, 32, 4, 2),  # 128 * 32 * 6 * 6
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2, 2),  # 128 * 32 * 3 * 3
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, 3, 1),  # 128 * 64 * 1 * 1
            nn.ReLU(inplace=True),

            nn.Flatten(),
        )

        self.fc1 = nn.Sequential(
            nn.Conv1d(57, 16, 4, 2, 1),  # 128 * 16 * 32
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(16, 32, 4, 2, 1),  # 128 * 32 * 8
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(4, 2, 1),  # 128 * 32 * 4
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, 4, 1),
            nn.ReLU(inplace=True),
            nn.Flatten()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(128, 9)
        )

        self.attention1 = Attention(in_channel=64)

        self.w1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.w2 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.w3 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.w1.data.fill_(1.0)
        self.w2.data.fill_(1.0)
        self.w3.data.fill_(1.0)

    def forward(self, x):
        x1 = x[0:self.batch_size, :, :, :]
        x2 = x[self.batch_size: 2 * self.batch_size, :, :, :]
        x3 = x[2 * self.batch_size: 3 * self.batch_size, :, :, :]

        encoder_x1 = self.encoder1(x1)
        encoder_x11 = encoder_x1.transpose(3, 1)
        encoder_x12 = encoder_x11.squeeze(2)  # 128 * 56 * 64

        encoder_x2 = self.encoder2(x2)
        encoder_x21 = encoder_x2.transpose(3, 1)
        encoder_x22 = encoder_x21.squeeze(2)  # 128 * 56 * 64

        encoder_x3 = self.encoder3(x3)  # 128 * 32 * 14 * 14

        decoder_x1 = self.decoder1(encoder_x1)
        decoder_x2 = self.decoder2(encoder_x2)
        decoder_x3 = self.decoder3(encoder_x3)

        encoder_x31 = self.conv(encoder_x3)  # 128 * 64
        encoder_x32 = encoder_x31.unsqueeze(1)

        out1 = torch.cat((encoder_x12, encoder_x22), dim=2)  # 时间上的信息拼接 128 * 56 * 128
        out2 = torch.cat((encoder_x22, encoder_x32), dim=1)  # 时空拼接 128 * 57 * 64

        lstm1, hid1 = self.lstm1(out1)  # 128 * 56 * 64
        out_12 = self.attention1(lstm1)  # 128 * 64

        out_13 = self.fc1(out2)  # 128 * 64

        out = torch.cat((out_12, out_13), dim=1)

        # out = torch.cat((out1, out2), dim=1)
        out = self.fc2(out)
        return decoder_x1, decoder_x2, decoder_x3, out


class NewModel12(nn.Module):
    def __init__(self, batch_size=128):
        super(NewModel12, self).__init__()
        self.batch_size = batch_size
        self.encoder1 = Encoder(mode="LSTM")
        self.encoder2 = Encoder(mode="LSTM")
        self.encoder3 = Encoder(mode="CNN")

        self.decoder1 = Decoder(mode="LSTM")
        self.decoder2 = Decoder(mode="LSTM")
        self.decoder3 = Decoder(mode="CNN")

        self.lstm1 = nn.GRU(input_size=128, hidden_size=64, num_layers=2, batch_first=True, bidirectional=False)
        self.lstm2 = nn.GRU(input_size=128, hidden_size=64, num_layers=2, batch_first=True, bidirectional=False)

        self.conv = nn.Sequential(
            nn.Conv2d(32, 32, 4, 2),  # 128 * 32 * 6 * 6
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2, 2),  # 128 * 32 * 3 * 3
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, 3, 1),  # 128 * 64 * 1 * 1
            nn.ReLU(inplace=True),

            nn.Flatten(),
        )

        self.fc1 = nn.Sequential(
            nn.Conv1d(57, 16, 4, 2, 1),  # 128 * 16 * 32
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(16, 32, 4, 2, 1),  # 128 * 32 * 8
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(4, 2, 1),  # 128 * 32 * 4
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, 4, 1),
            nn.ReLU(inplace=True),
            nn.Flatten()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(64, 9)
        )

        self.attention1 = Attention(in_channel=64)

        self.w1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.w2 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.w1.data.fill_(1.0)
        self.w2.data.fill_(1.0)

    def forward(self, x):
        x1 = x[0:self.batch_size, :, :, :]
        x2 = x[self.batch_size: 2 * self.batch_size, :, :, :]
        x3 = x[2 * self.batch_size: 3 * self.batch_size, :, :, :]

        encoder_x1 = self.encoder1(x1)
        encoder_x11 = encoder_x1.transpose(3, 1)
        encoder_x12 = encoder_x11.squeeze(2)  # 128 * 56 * 64

        encoder_x2 = self.encoder2(x2)
        encoder_x21 = encoder_x2.transpose(3, 1)
        encoder_x22 = encoder_x21.squeeze(2)  # 128 * 56 * 64

        encoder_x3 = self.encoder3(x3)  # 128 * 32 * 14 * 14

        decoder_x1 = self.decoder1(encoder_x1)
        decoder_x2 = self.decoder2(encoder_x2)
        decoder_x3 = self.decoder3(encoder_x3)

        encoder_x31 = self.conv(encoder_x3)  # 128 * 64
        encoder_x32 = encoder_x31.unsqueeze(1)

        out1 = torch.cat((encoder_x12, encoder_x22), dim=2)  # 时间上的信息拼接 128 * 56 * 128
        out2 = torch.cat((encoder_x12, encoder_x32), dim=1)  # 时空拼接 128 * 57 * 64

        lstm1, hid1 = self.lstm1(out1)  # 128 * 56 * 64
        out_12 = self.attention1(lstm1)  # 128 * 64

        out_13 = self.fc1(out2)  # 128 * 64

        # out = torch.cat((out_12, out_13), dim=1)

        out = self.w1 * out_12 + self.w2 * out_13
        out = self.fc2(out)
        return decoder_x1, decoder_x2, decoder_x3, out


class NewModel13(nn.Module):
    def __init__(self, batch_size=128):
        super(NewModel13, self).__init__()
        self.batch_size = batch_size
        self.encoder1 = Encoder(mode="LSTM")
        self.encoder2 = Encoder(mode="LSTM")
        self.encoder3 = Encoder(mode="CNN")

        self.decoder1 = Decoder(mode="LSTM")
        self.decoder2 = Decoder(mode="LSTM")
        self.decoder3 = Decoder(mode="CNN")

        self.lstm1 = nn.GRU(input_size=128, hidden_size=64, num_layers=2, batch_first=True, bidirectional=False)
        self.lstm2 = nn.GRU(input_size=128, hidden_size=64, num_layers=2, batch_first=True, bidirectional=False)

        self.conv = nn.Sequential(
            nn.Conv2d(32, 32, 4, 2),  # 128 * 32 * 6 * 6
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2, 2),  # 128 * 32 * 3 * 3
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, 3, 1),  # 128 * 64 * 1 * 1
            nn.ReLU(inplace=True),

            nn.Flatten(),
        )

        self.fc1 = nn.Sequential(
            nn.Conv1d(57, 16, 4, 2, 1),  # 128 * 16 * 32
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(16, 32, 4, 2, 1),  # 128 * 32 * 8
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(4, 2, 1),  # 128 * 32 * 4
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, 4, 1),
            nn.ReLU(inplace=True),
            nn.Flatten()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(64, 9)
        )

        self.attention1 = Attention(in_channel=64)

        self.w1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.w2 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.w1.data.fill_(1.0)
        self.w2.data.fill_(1.0)

    def forward(self, x):
        x1 = x[0:self.batch_size, :, :, :]
        x2 = x[self.batch_size: 2 * self.batch_size, :, :, :]
        x3 = x[2 * self.batch_size: 3 * self.batch_size, :, :, :]

        encoder_x1 = self.encoder1(x1)
        encoder_x11 = encoder_x1.transpose(3, 1)
        encoder_x12 = encoder_x11.squeeze(2)  # 128 * 56 * 64

        encoder_x2 = self.encoder2(x2)
        encoder_x21 = encoder_x2.transpose(3, 1)
        encoder_x22 = encoder_x21.squeeze(2)  # 128 * 56 * 64

        encoder_x3 = self.encoder3(x3)  # 128 * 32 * 14 * 14

        decoder_x1 = self.decoder1(encoder_x1)
        decoder_x2 = self.decoder2(encoder_x2)
        decoder_x3 = self.decoder3(encoder_x3)

        encoder_x31 = self.conv(encoder_x3)  # 128 * 64
        encoder_x32 = encoder_x31.unsqueeze(1)

        out1 = torch.cat((encoder_x12, encoder_x22), dim=2)  # 时间上的信息拼接 128 * 56 * 128
        out2 = torch.cat((encoder_x22, encoder_x32), dim=1)  # 时空拼接 128 * 57 * 64

        lstm1, hid1 = self.lstm1(out1)  # 128 * 56 * 64
        out_12 = self.attention1(lstm1)  # 128 * 64

        out_13 = self.fc1(out2)  # 128 * 64

        # out = torch.cat((out_12, out_13), dim=1)

        out = self.w1 * out_12 + self.w2 * out_13
        out = self.fc2(out)
        return decoder_x1, decoder_x2, decoder_x3, out