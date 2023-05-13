import numpy as np
import torch
import torch.nn as nn

# original link : https://www.sciencedirect.com/science/article/pii/S092523122031780X

class DCNN_Block(nn.Module):
    def __init__(self):
        super(DCNN_Block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 6, 5, 1, 2),  # 6 * 224 * 224
            nn.MaxPool2d(2, 2),  # 6 * 112 * 112
            nn.ReLU(inplace=True),

            nn.Conv2d(6, 12, 5, 1, 2),  # 12 * 112 * 112
            nn.MaxPool2d(2, 2),  # 12 * 56 * 56
            nn.ReLU(inplace=True),

            nn.Conv2d(12, 24, 5, 1, 2),  # 24 * 56 * 56
            nn.MaxPool2d(2, 2),  # 24 * 28 * 28
            nn.ReLU(inplace=True),

            nn.Conv2d(24, 48, 5, 1, 2),  # 48 * 28 * 28
            nn.MaxPool2d(2, 2),  # 48 * 14 * 14
            nn.ReLU(inplace=True),

            nn.Conv2d(48, 48, 5, 1, 2),  # 48 * 14 * 14
            nn.MaxPool2d(2, 2),  # 48 * 7 * 7
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)

        return x


class DCNN_MFS(nn.Module):
    def __init__(self):
        super(DCNN_MFS, self).__init__()

        self.branch1 = DCNN_Block()
        self.branch2 = DCNN_Block()
        self.branch3 = DCNN_Block()

        self.conv = nn.Sequential(
            nn.Conv2d(144, 48, 1),  # batch * 48 * 7 * 7
            nn.ReLU(inplace=True),
        )

        self.fc = nn.Sequential(
            nn.Linear(2352, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        tr, td, cvd = x.chunk(3, dim=0)

        tr_feature = self.branch1(tr)  # batch * 48 * 7 * 7
        td_feature = self.branch2(td)  # batch * 48 * 7 * 7
        cvd_feature = self.branch3(cvd)  # batch * 48 * 7 * 7
        feature_vec = torch.cat((tr_feature, td_feature, cvd_feature), dim=1)  # batch * 144 * 7 * 7
        feature_vec = self.conv(feature_vec)

        feature_vec = feature_vec.view(feature_vec.shape[0], -1)
        cls_out = self.fc(feature_vec)

        return cls_out

    def predict(self, x):
        tr, td, cvd = x.chunk(3, dim=0)

        tr_feature = self.branch1(tr)  # batch * 48 * 6 * 6
        td_feature = self.branch2(td)  # batch * 48 * 6 * 6
        cvd_feature = self.branch3(cvd)  # batch * 48 * 6 * 6
        feature_vec = torch.cat((tr_feature, td_feature, cvd_feature), dim=1)  # batch * 144 * 7 * 7

        feature_vec = self.conv(feature_vec)

        feature_vec = feature_vec.view(feature_vec.shape[0], -1)
        cls_out = self.fc(feature_vec)

        return cls_out

