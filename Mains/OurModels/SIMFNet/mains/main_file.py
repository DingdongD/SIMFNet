import sys, os

sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))

import time
from utils import LoadData, WriteData, Params, VisualizeData
from sup_models import DCNN_MFS, GMFN, MDFRadarNet, TSFNet, LabelSmoothing
from sup_train import sup1_train, sup2_train

import torch.optim as optim
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
# from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from torch.utils.data import random_split, DataLoader
import numpy as np

# random seed
if Params.seed_flag is not None:
    np.random.seed(Params.seed_flag)
    torch.manual_seed(Params.seed_flag)
    torch.cuda.manual_seed(Params.seed_flag)
    cudnn.deterministic = True

cudnn.benchmark = True  # to enhance efficiency


# root_dir = ['/home/yuxl/xl_project/BaseDataset/Domain_2']  # BaseDataset

# root_dir = ['/home/djf/xl_project/BaseDataset/Domain_1', '/home/djf/xl_project/BaseDataset/Domain_5']

root_dir = ['/home/yuxl/xl_project/BaseDataset/Domain_1', '/home/yuxl/xl_project/BaseDataset/Domain_3', '/home/yuxl/xl_project/BaseDataset/Domain_5']

test_dir = ['/home/yuxl/xl_project/BaseDataset/Domain_2']

# dataset split
tr_train_dataset = LoadData.CustomDataset(root_dir=root_dir, transform=True, view_type='TR', data_type='train')
tr_val_dataset = LoadData.CustomDataset(root_dir=root_dir, transform=True, view_type='TR', data_type='val')
tr_test_dataset = LoadData.CustomDataset1(root_dir=test_dir, transform=True, view_type='TR', data_type='test')

td_train_dataset = LoadData.CustomDataset(root_dir=root_dir, transform=True, view_type='TD', data_type='train')
td_val_dataset = LoadData.CustomDataset(root_dir=root_dir, transform=True, view_type='TD', data_type='val')
td_test_dataset = LoadData.CustomDataset1(root_dir=test_dir, transform=True, view_type='TD', data_type='test')

cvd_train_dataset = LoadData.CustomDataset(root_dir=root_dir, transform=True, view_type='CVD', data_type='train')
cvd_val_dataset = LoadData.CustomDataset(root_dir=root_dir, transform=True, view_type='CVD', data_type='val')
cvd_test_dataset = LoadData.CustomDataset1(root_dir=test_dir, transform=True, view_type='CVD', data_type='test')

train_dataset = {'tr': tr_train_dataset, 'td': td_train_dataset, 'cvd': cvd_train_dataset}
val_dataset = {'tr': tr_val_dataset, 'td': td_val_dataset, 'cvd': cvd_val_dataset}
test_dataset = {'tr': tr_test_dataset, 'td': td_test_dataset, 'cvd': cvd_test_dataset}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = GMFN.GMFN()
#model = DCNN_MFS.DCNN_MFS()
# model = MDFRadarNet.MDFRadarNet()

# model = TSFNet.TSFNet()
model = ADGCN.ADGCN()
if Params.use_gpu:
    model.to(device)
    
# optimizer = optim.SGD(model.parameters(), lr=Params.learning_rate, momentum=Params.momentum, weight_decay=Params.weight_decay)
optimizer = optim.Adam(model.parameters(), lr=Params.learning_rate, weight_decay=Params.weight_decay)
lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda x: Params.learning_rate * (1. + Params.lr_gamma *
                                           float(x)) ** (-Params.lr_decay))
cls_criterion = LabelSmoothing.labelsmoothing()
# cls_criterion = nn.CrossEntropyLoss()

def my_main_1(save_path): # designed for other multi-domain fusion models
    indices1 = torch.randperm(len(tr_train_dataset))
    mySampler1 = LoadData.SamplerDef(indices=indices1)
    indices2 = torch.randperm(len(tr_val_dataset))
    mySampler2 = LoadData.SamplerDef(indices=indices2)
    train_dataloader = LoadData.get_dataloader(train_dataset, batch_size=Params.train_batch_size, joint_idx=mySampler1)
    val_dataloader = LoadData.get_dataloader(val_dataset, batch_size=Params.train_batch_size, joint_idx=mySampler2)
    
    train_hist = {'train_acc': [], 'train_loss': []}
    val_hist = {'val_acc': [], 'val_loss': []}
    best_acc = 0.
    count_overfitting = 0

    for epoch in range(Params.epochs):
        t0 = time.time()
        print("Epoch:{}".format(epoch))
        sup1_train.train(train_dataloader, model, device, cls_criterion, optimizer, lr_scheduler, train_hist)
        WriteData.write_data_to_csv(save_path, train_hist, data_type='train', epoch=epoch)
        t1 = time.time() - t0
        sup1_train.validate(val_dataloader, model, device, cls_criterion, val_hist)
        WriteData.write_data_to_csv(save_path, val_hist, data_type='val', epoch=epoch)
        print("Training Time Cost:{:.4f}s".format(t1))

        if val_hist['val_acc'][-1] > best_acc:
            best_acc = val_hist['val_acc'][-1]
            print("---- Best Accuracy: {:.4f} ----".format(best_acc))
            torch.save(model, save_path + 'model.pth')
            count_overfitting = 0
        else:
            count_overfitting += 1

        if count_overfitting == Params.overfitting_threshold:
            break
            
            
def my_main_2(save_path): # designed for TSFNet
    indices1 = torch.randperm(len(tr_train_dataset))
    mySampler1 = LoadData.SamplerDef(indices=indices1)
    indices2 = torch.randperm(len(tr_val_dataset))
    mySampler2 = LoadData.SamplerDef(indices=indices2)
    train_dataloader = LoadData.get_dataloader(train_dataset, batch_size=Params.train_batch_size, joint_idx=mySampler1)
    val_dataloader = LoadData.get_dataloader(val_dataset, batch_size=Params.train_batch_size, joint_idx=mySampler2)

    train_hist = {'train_acc': [], 'train_loss': []}
    val_hist = {'val_acc': [], 'val_loss': []}
    best_acc = 0.
    count_overfitting = 0

    for epoch in range(Params.epochs):
        t0 = time.time()
        print("Epoch:{}".format(epoch))
        sup2_train.train(train_dataloader, model, device, cls_criterion, optimizer, lr_scheduler, train_hist)
        WriteData.write_data_to_csv(save_path, train_hist, data_type='train', epoch=epoch)
        t1 = time.time() - t0
        sup1_train.validate(val_dataloader, model, device, cls_criterion, val_hist)
        WriteData.write_data_to_csv(save_path, val_hist, data_type='val', epoch=epoch)
        print("Training Time Cost:{:.4f}s".format(t1))

        if val_hist['val_acc'][-1] > best_acc:
            best_acc = val_hist['val_acc'][-1]
            print("---- Best Accuracy: {:.4f} ----".format(best_acc))
            torch.save(model, save_path + 'model.pth')
            count_overfitting = 0
        else:
            count_overfitting += 1

        if count_overfitting == Params.overfitting_threshold:
            break


# write predicted results into csv
def write_result_csv(model_path, csv_path):
    model_path += 'model.pth'
    model = torch.load(model_path)

    train_dataset = {'tr': tr_train_dataset, 'td': td_train_dataset, 'cvd': cvd_train_dataset}
    test_dataset = {'tr': tr_test_dataset, 'td': td_test_dataset, 'cvd': cvd_test_dataset}

    train_indices = torch.randperm(len(tr_train_dataset))
    test_indices = torch.randperm(len(tr_test_dataset))
    mySampler1 = LoadData.SamplerDef(indices=train_indices)
    mySampler2 = LoadData.SamplerDef(indices=test_indices)

    src_test_dataloader = LoadData.get_dataloader(train_dataset, batch_size=1, joint_idx=mySampler1)
    tgt_test_dataloader = LoadData.get_dataloader(test_dataset, batch_size=1, joint_idx=mySampler2)

    pred_hist = {'image_path': [], 'real_label': [], 'predict_label': [], 'logit_vector': []}
    sup1_train.predict(src_test_dataloader, model, device, pred_hist)
    WriteData.write_data_to_csv(csv_path, pred_hist, data_type='test', epoch='train')

    pred_hist = {'image_path': [], 'real_label': [], 'predict_label': [], 'logit_vector': []}
    sup1_train.predict(tgt_test_dataloader, model, device, pred_hist)
    WriteData.write_data_to_csv(csv_path, pred_hist, data_type='test', epoch='test')



if __name__ == '__main__':
    # save_path = '/home/djf/xl_project/Mycode/logs/ADGCN/6/'  # MDFRadarNet
    # save_path = '/home/djf/xl_project/Mycode/base_logs/DCNN_MFS/2/'  # MDFRadarNet
    # save_path = '/home/yuxl/xl_project/Mycode/test_logs/DCNNMFS/135/'  # MDFRadarNet
    save_path = '/home/yuxl/xl_project/Mycode/abl_logs/ADGCN/135_5_4/'  # MDFRadarNet
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    #my_main_1(save_path)
    # write_result_csv(save_path, save_path)
    # VisualizeData.supervised_tsne(train_dataset, test_dataset, save_path)
    #VisualizeData.supervised_tsne1(test_dataset, save_path)
    VisualizeData.supervised_cmap(test_dataset, save_path)
    # model = torch.load("/home/djf/xl_project/Mycode/logs/ADGCN/1/model.pth")
    # print(model)

