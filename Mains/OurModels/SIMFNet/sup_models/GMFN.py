import torch
import torch.nn as nn
import numpy as np
from torch_geometric.nn import GCNConv, global_sort_pool
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import scipy.sparse as sp

# original link GRS：https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9777687

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MFEsN(nn.Module):
    def __init__(self):
        super(MFEsN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 6, 5, 2, 1),  # 6 * 111 * 111
            nn.BatchNorm2d(6),
            nn.ReLU(inplace=True),

            nn.Conv2d(6, 12, 5, 2, 1),  # 12 * 55 * 55
            nn.BatchNorm2d(12),
            nn.ReLU(inplace=True),

            nn.Conv2d(12, 24, 5, 2, 1),  # 24 * 27 * 27
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),

            nn.Conv2d(24, 48, 5, 2, 1),  # 48 * 13 * 13
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),

            nn.Conv2d(48, 48, 5, 2, 1),  # 48 * 6 * 6
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )

        self.fc = nn.Flatten()

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)

        return x


class MFGsN(nn.Module):
    def __init__(self):
        super(MFGsN, self).__init__()

        self.gconv1 = GCNConv(in_channels=1728, out_channels=512)
        self.relu1 = nn.ReLU(inplace=True)
        self.gconv2 = GCNConv(in_channels=512, out_channels=256)
        self.relu2 = nn.ReLU(inplace=True)
        
    def forward(self, x, edge_index, edge_weight):
        conv_x = self.relu1(self.gconv1(x, edge_index, edge_weight))
        out = self.relu2(self.gconv2(conv_x, edge_index, edge_weight))

        return out


class GCsN(nn.Module):
    def __init__(self):
        super(GCsN, self).__init__()
        self.fc0 = nn.Flatten()
        self.fc1 = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 10),
            # nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.fc0(x)
        x = self.fc1(x)

        return x


class GMFN(nn.Module):
    def __init__(self):
        super(GMFN, self).__init__()
        # CNN extractor part
        self.MFEsN1 = MFEsN()
        self.MFEsN2 = MFEsN()
        self.MFEsN3 = MFEsN()
        # GCN construct graph part
        self.MFGsN = MFGsN()
        # graph classification part
        self.GCsN = GCsN()

        # edge and weight construct
        A = np.array([[1, 1, 1],
                      [1, 1, 1],
                      [1, 1, 1]])
        edge_index_temp = sp.coo_matrix(A)
        edge_index = np.vstack((edge_index_temp.row, edge_index_temp.col))
        edge_index = torch.LongTensor(edge_index)
        edge_weight = torch.FloatTensor(edge_index_temp.data)

        self.edge_index = edge_index.to(device)
        self.edge_weight = edge_weight.to(device)

    def forward(self, x):
        tr, td, cvd = x.chunk(3, dim=0)  # 4 * 1 * 224 * 224
        tr_feature = self.MFEsN1(tr).unsqueeze(1)
        td_feature = self.MFEsN2(td).unsqueeze(1)
        cvd_feature = self.MFEsN3(cvd).unsqueeze(1)
        gcn_input = torch.cat((tr_feature, td_feature, cvd_feature), dim=1)  # batch * 3 * feature_dim

        gcn_feature = self.MFGsN(gcn_input, self.edge_index, self.edge_weight)  # batch * 3 * 256

        gcn_x = []
        for batch_idx in range(gcn_feature.shape[0]):
            gcn_x.append(Data(x=gcn_feature[batch_idx], edge_index=self.edge_index))
        gcn_loader = DataLoader(gcn_x, batch_size=gcn_feature.shape[0])
        for data in gcn_loader:
            out = global_sort_pool(data.x, data.batch, k=3)

        gcn_out = self.GCsN(out)

        return gcn_out

    def predict(self, x):
        tr, td, cvd = x.chunk(3, dim=0)  # 4 * 1 * 224 * 224
        tr_feature = self.MFEsN1(tr).unsqueeze(1)
        td_feature = self.MFEsN2(td).unsqueeze(1)
        cvd_feature = self.MFEsN3(cvd).unsqueeze(1)
        gcn_input = torch.cat((tr_feature, td_feature, cvd_feature), dim=1)  # batch * 3 * feature_dim

        gcn_feature = self.MFGsN(gcn_input, self.edge_index, self.edge_weight)  # batch * 3 * 256

        gcn_x = []
        for batch_idx in range(gcn_feature.shape[0]):
            gcn_x.append(Data(x=gcn_feature[batch_idx], edge_index=self.edge_index))
        gcn_loader = DataLoader(gcn_x, batch_size=gcn_feature.shape[0])
        for data in gcn_loader:
            out = global_sort_pool(data.x, data.batch, k=3)

        gcn_out = self.GCsN(out)

        return gcn_out

