import numpy as np
import torch
import torch.nn as nn

from torchsummary import summary


# original link TAES£ºhttps://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9385952


class Conv1DLSTM(nn.Module):
    def __init__(self):
        super(Conv1DLSTM, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 1, 1),  # 1 * 224 * 224
            nn.ReLU(inplace=True),
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(224, 128, 5, 1, 2),  # batch * 32 * 224
            nn.MaxPool1d(2, 2),
            nn.ReLU(inplace=True),

            nn.Conv1d(128, 128, 5, 1, 2),  # batch * 64 * 112
            nn.ReLU(inplace=True),

        )

        self.lstm = nn.LSTM(input_size=128, hidden_size=128, num_layers=1, batch_first=True, bidirectional=True)

    def forward(self, x):
        x = self.conv1(x).squeeze(1)
        x = self.conv2(x)
        x = x.transpose(2, 1)

        out, _ = self.lstm(x)  # batch * 112 * 64
        return out[:, -1, :]


class Conv2D(nn.Module):
    def __init__(self):
        super(Conv2D, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 8, 5, 1, 2),
            nn.MaxPool2d(4, 4),  # 8 * 56 * 56
            nn.ReLU(inplace=True),


            nn.Conv2d(8, 16, 5, 1, 2),  # 16 * 56 * 56
            nn.MaxPool2d(4, 4),  # 16 * 14 * 14
            nn.ReLU(inplace=True),


            nn.Conv2d(16, 32, 5, 1, 2),
            nn.MaxPool2d(4, 4),  # 32 * 3 * 3
            nn.ReLU(inplace=True),


            nn.Conv2d(32, 64, 2),  # 64 * 2 * 2
            nn.ReLU(inplace=True),
        )

        self.fc = nn.Flatten()

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)

        return x


class MDFRadarNet(nn.Module):
    def __init__(self):
        super(MDFRadarNet, self).__init__()
        self.branch1 = Conv1DLSTM()
        self.branch2 = Conv1DLSTM()
        self.branch3 = Conv2D()
        self.fc = nn.Sequential(
            nn.Linear(256, 10),
        )

        self.w1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.w2 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.w3 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.w1.data.fill_(1.0)
        self.w2.data.fill_(1.0)
        self.w3.data.fill_(1.0)

    def forward(self, x):
        tr, td, cvd = x.chunk(3, dim=0)
        tr_feature = self.branch1(tr)
        td_feature = self.branch2(td)
        cvd_feature = self.branch3(cvd)

        out = self.w1 * tr_feature + self.w2 * td_feature + self.w3 * cvd_feature
        out = self.fc(out)
        return out

    def predict(self, x):
        tr, td, cvd = x.chunk(3, dim=0)
        tr_feature = self.branch1(tr)
        td_feature = self.branch2(td)
        cvd_feature = self.branch3(cvd)

        out = self.w1 * tr_feature + self.w2 * td_feature + self.w3 * cvd_feature
        out = self.fc(out)
        return out


"""
model = torch.load('H:\MyDataset\SupervisedDemo\logs\model.pth')
print(model)
"""