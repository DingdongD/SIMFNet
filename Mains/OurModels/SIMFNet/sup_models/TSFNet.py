import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# original link ICARCV：https://ieeexplore.ieee.org/document/10004245

class Attention(nn.Module):
    def __init__(self, in_channel: int):
        super(Attention, self).__init__()
        self.w = nn.Linear(in_channel, 1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, H):
        M = self.tanh(H)  # (batch, seq_len, rnn_size)
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
            nn.Conv2d(3, 16, (8, 4), (4, 2), (2, 1)),  # batch * 16 * 56 * 112
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 32, (8, 4), (4, 2), (2, 1)),  # batch * 32 * 14 * 56
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 48, (4, 3), (2, 1), (1, 1)),  # batch * 48 * 7  * 56
            nn.ReLU(inplace=True),

            nn.Conv2d(48, 64, (4, 3), (1, 1), (0, 1)),  # batch * 64 * 4 * 56
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, (4, 3), (1, 1), (0, 1)),  # batch * 64 * 1 * 56
            nn.ReLU(inplace=True),

        )

        self.encoder2 = nn.Sequential(
            nn.Conv2d(3, 8, 8, 4, 2),  # 128 * 8 * 56 * 56
            nn.ReLU(inplace=True),

            nn.Conv2d(8, 16, 8, 4, 2),  # 128 * 16 * 14 * 14
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 32, 4, 2, 1),  # 128 * 32 * 7 * 7
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, 4, 1),  # 128 * 64 * 4 * 4
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, 4, 1),  # 128 * 64 * 1 * 1
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
            nn.ConvTranspose2d(64, 64, (4, 3), (1, 1), (0, 1)),  # 128 * 64 * 4 * 56
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 48, (4, 3), (1, 1), (0, 1)),  # 128 * 48 * 7 * 56
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(48, 32, (4, 3), (2, 1), (1, 1)),  # 128 * 32 * 14 * 56
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(32, 16, (8, 4), (4, 2), (2, 1)),  # 128 * 16 * 56 * 112
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(16, 3, (8, 4), (4, 2), (2, 1)),  # 128 * 3 * 224 * 224
            nn.Sigmoid(),
        )

        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 4, 1),  # 128 * 64 * 3 * 3
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 1),  # 128 * 32 * 7 * 7
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, 4, 2, 1),  # 128 * 16 * 14 * 14
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 8, 8, 4, 2),  # 128 * 16 * 56 * 56
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(8, 3, 8, 4, 2),  # 128 * 3 * 224 * 224
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        if self.mode == "LSTM":
            out = self.decoder1(x)
        elif self.mode == "CNN":
            out = self.decoder2(x)
        return out


class TSFNet(nn.Module):
    def __init__(self):
        super(TSFNet, self).__init__()
        self.encoder1 = Encoder(mode="LSTM")
        self.encoder2 = Encoder(mode="LSTM")
        self.encoder3 = Encoder(mode="CNN")

        self.decoder1 = Decoder(mode="LSTM")
        self.decoder2 = Decoder(mode="LSTM")
        self.decoder3 = Decoder(mode="CNN")

        self.lstm1 = nn.GRU(input_size=128, hidden_size=64, num_layers=2, batch_first=True, bidirectional=False)

        self.conv = nn.Sequential(
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
            nn.Linear(64, 10)
        )

        self.attention1 = Attention(in_channel=64)

        self.w1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.w2 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.w1.data.fill_(1.0)
        self.w2.data.fill_(1.0)

        self.rc_loss = nn.MSELoss()

    def forward(self, x):
        x1, x2, x3 = x.chunk(3, dim=0)

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

        out1 = torch.cat((encoder_x12, encoder_x22), dim=2)  # 128 * 56 * 128
        out2 = torch.cat((encoder_x12, encoder_x32), dim=1)  # 128 * 57 * 64

        lstm1, hid1 = self.lstm1(out1)  # 128 * 56 * 64
        out_12 = self.attention1(lstm1)  # 128 * 64

        out_13 = self.fc1(out2)  # 128 * 64

        # out = torch.cat((out_12, out_13), dim=1)

        out = self.w1 * out_12 + self.w2 * out_13
        out = self.fc2(out)
        rc_loss = self.rc_loss(x1, decoder_x1) + self.rc_loss(x2, decoder_x2) + self.rc_loss(x3, decoder_x3)

        return out, rc_loss

    def predict(self, x):
        x1, x2, x3 = x.chunk(3, dim=0)

        encoder_x1 = self.encoder1(x1)
        encoder_x11 = encoder_x1.transpose(3, 1)
        encoder_x12 = encoder_x11.squeeze(2)  # 128 * 56 * 64

        encoder_x2 = self.encoder2(x2)
        encoder_x21 = encoder_x2.transpose(3, 1)
        encoder_x22 = encoder_x21.squeeze(2)  # 128 * 56 * 64

        encoder_x3 = self.encoder3(x3)  # 128 * 32 * 14 * 14

        encoder_x31 = self.conv(encoder_x3)  # 128 * 64
        encoder_x32 = encoder_x31.unsqueeze(1)

        out1 = torch.cat((encoder_x12, encoder_x22), dim=2)  # 128 * 56 * 128
        out2 = torch.cat((encoder_x12, encoder_x32), dim=1)  # 128 * 57 * 64

        lstm1, hid1 = self.lstm1(out1)  # 128 * 56 * 64
        out_12 = self.attention1(lstm1)  # 128 * 64

        out_13 = self.fc1(out2)  # 128 * 64

        # out = torch.cat((out_12, out_13), dim=1)

        out = self.w1 * out_12 + self.w2 * out_13
        out = self.fc2(out)

        return out

