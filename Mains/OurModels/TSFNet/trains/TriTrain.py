import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms
import numpy as np
import torch.nn.functional as F

import TriModels
from CustomDataset import CustomDataset, SamplerDef
from torch.autograd import Variable
import matplotlib.pyplot as plt

# 定义超参数
batch_size = 128
learning_rate = 1e-3
num_classes = 9

# 不同输入文件的路径
train_rt_path = "H:/paper_code/fusionnet/Data_v3/train_rt/"
train_md_path = "H:/paper_code/fusionnet/Data_v3/train_md/"
train_cvd_path = "H:/paper_code/fusionnet/Data_v3/train_cvd/"
test_rt_path = "H:/paper_code/fusionnet/Data_v3/test_rt/"
test_md_path = "H:/paper_code/fusionnet/Data_v3/test_md/"
test_cvd_path = "H:/paper_code/fusionnet/Data_v3/test_cvd/"

# 加载数据集
train_rt_dataset = CustomDataset(root_dir=train_rt_path, transform=True)
train_md_dataset = CustomDataset(root_dir=train_md_path, transform=True)
train_cvd_dataset = CustomDataset(root_dir=train_cvd_path, transform=True)
test_rt_dataset = CustomDataset(root_dir=test_rt_path, transform=True)
test_md_dataset = CustomDataset(root_dir=test_md_path, transform=True)
test_cvd_dataset = CustomDataset(root_dir=test_cvd_path, transform=True)

train_dataloader = {}
n = len(train_rt_dataset)
indices = torch.randperm(n)
mySampler = SamplerDef(indices=indices)
train_dataloader['v1'] = DataLoader(dataset=train_rt_dataset, batch_size=batch_size, shuffle=False, pin_memory=True,
                                    sampler=mySampler, drop_last=True, num_workers=4)
train_dataloader['v2'] = DataLoader(dataset=train_md_dataset, batch_size=batch_size, shuffle=False, pin_memory=True,
                                    sampler=mySampler, drop_last=True, num_workers=4)
train_dataloader['v3'] = DataLoader(dataset=train_cvd_dataset, batch_size=batch_size, shuffle=False, pin_memory=True,
                                    sampler=mySampler, drop_last=True, num_workers=4)

test_dataloader = {}
n = len(test_rt_dataset)
indices = torch.randperm(n)
mySampler = SamplerDef(indices=indices)
test_dataloader['v1'] = DataLoader(dataset=test_rt_dataset, batch_size=batch_size, shuffle=False, pin_memory=True,
                                   sampler=mySampler, drop_last=True, num_workers=4)
test_dataloader['v2'] = DataLoader(dataset=test_md_dataset, batch_size=batch_size, shuffle=False, pin_memory=True,
                                   sampler=mySampler, drop_last=True, num_workers=4)
test_dataloader['v3'] = DataLoader(dataset=test_cvd_dataset, batch_size=batch_size, shuffle=False, pin_memory=True,
                                   sampler=mySampler, drop_last=True, num_workers=4)


def train_model(train_dataloader, test_dataloader, model, batch_size, num_epoch, loss_fn,
                optimizer, path_name, idx):
    train_loss = []
    test_loss = []
    train_acc = []
    test_acc = []
    avg_train_loss = []
    avg_test_loss = []
    avg_train_acc = []
    avg_test_acc = []
    count_overfitting = 0
    for epoch in range(1, num_epoch + 1):
        model.train()
        for (X, y) in enumerate(zip(train_dataloader['v1'], train_dataloader['v2'], train_dataloader['v3'])):
            input_data = torch.cat((y[0][0], y[1][0], y[2][0]), dim=0)
            input_data = input_data.cuda()
            output_data = torch.as_tensor(y[0][1])
            output_data = output_data.cuda()

            optimizer.zero_grad()
            output = model(input_data)
            loss = loss_fn(output, output_data)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            train_acc.append((output.argmax(1) == output_data).type(torch.float).sum().item() / batch_size)

        model.eval()
        for (X, y) in enumerate(zip(test_dataloader['v1'], test_dataloader['v2'], test_dataloader['v3'])):
            input_data = torch.cat((y[0][0], y[1][0], y[2][0]), dim=0)
            input_data = input_data.cuda()
            output_data = torch.as_tensor(y[0][1])
            output_data = output_data.cuda()

            output = model(input_data)
            loss = loss_fn(output, output_data)
            test_loss.append(loss.item())
            test_acc.append((output.argmax(1) == output_data).type(torch.float).sum().item() / batch_size)

        a_train_loss = np.average(train_loss)
        a_test_loss = np.average(test_loss)
        a_train_acc = np.average(train_acc)
        a_test_acc = np.average(test_acc)

        avg_train_loss.append(a_train_loss)
        avg_test_loss.append(a_test_loss)
        avg_train_acc.append(a_train_acc)
        avg_test_acc.append(a_test_acc)

        if a_test_loss <= np.min(avg_test_loss):
            print("=" * 100)
            print("Epoch " + str(epoch) + " is current best! valid acc: " + str(a_test_acc))
            final_path = path_name + str("best_model_%d.pth" % idx)
            torch.save(model, final_path)
            minposs = epoch
            print("=" * 100)
            count_overfitting = 0
        else:
            count_overfitting += 1

        if count_overfitting == 30:
            break

        epoch_len = len(str(num_epoch))
        print_msg = (f'[{epoch:>{epoch_len}}/{num_epoch:>{epoch_len}}] ' +
                     f'train_loss: {a_train_loss:.5f} ' +
                     f'test_loss: {a_test_loss:.5f} ' +
                     f'train_acc: {a_train_acc * 100:.5f}% ' +
                     f'test_acc: {a_test_acc * 100:.5f}% ')
        print(print_msg)

        train_loss = []
        test_loss = []
        train_acc = []
        test_acc = []

    return model, avg_train_loss, avg_test_loss, avg_train_acc, avg_test_acc, minposs

def draw_fig(list1, list2, name, minposs, path, idx):
    x1 = range(1, len(list1) + 1)
    y1 = list1
    y2 = list2
    if name == "loss":
        path = path + str("loss_%d.png" % idx)
        plt.cla()
        plt.title('Train loss vs. epoch', fontsize=20)
        plt.plot(x1, y1, '.-', label='Training Loss')
        plt.plot(x1, y2, '-*', label='Test Loss')
        plt.axvline(minposs, linestyle='--', color='r', label='Early Stopping Checkpoint')
        plt.xlabel('Epoch', fontsize=20)
        plt.ylabel('Loss', fontsize=20)
        plt.legend()
        plt.tight_layout()
        plt.grid()
        plt.savefig(path)
        plt.show()
    elif name == "acc":
        path = path + str("accuracy_%d.png" % idx)
        plt.cla()
        plt.title('Train accuracy vs. epoch', fontsize=20)
        plt.plot(x1, y1, '.-', label='Training Accuracy')
        plt.plot(x1, y2, '-*', label='Test Accuracy')
        plt.axvline(minposs, linestyle='--', color='r', label='Early Stopping Checkpoint')
        plt.xlabel('Epoch', fontsize=20)
        plt.ylabel('Accuracy', fontsize=20)
        plt.legend()
        plt.tight_layout()
        plt.grid()
        plt.savefig(path)
        plt.show()


def write_to_txt(txt_name, data):
    file = open(txt_name, 'w')
    for i in range(len(data)):
        file.write(str(data[i]))
        file.write('\n')
    file.close()


if __name__ == "__main__":
    '''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = TriModels.TriCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    num_epochs = 200
    loss_fn = nn.CrossEntropyLoss()
    path_name = './log2/tricnn1/'
    idx = 4

    model, avg_train_loss, avg_test_loss, avg_train_acc, avg_test_acc, minposs = train_model(train_dataloader,
                                                                                             test_dataloader,
                                                                                             model,
                                                                                             batch_size, num_epochs,
                                                                                             loss_fn,
                                                                                             optimizer, path_name, idx)
    train_acc_txt_name = path_name + str("train_acc_%d.txt" % idx)
    valid_acc_txt_name = path_name + str("test_acc_%d.txt" % idx)
    train_loss_txt_name = path_name + str("train_loss_%d.txt" % idx)
    valid_loss_txt_name = path_name + str("test_loss_%d.txt" % idx)

    write_to_txt(train_acc_txt_name, avg_train_acc)  # 保存 loss和acc文件
    write_to_txt(valid_acc_txt_name, avg_test_acc)
    write_to_txt(train_loss_txt_name, avg_train_loss)
    write_to_txt(valid_loss_txt_name, avg_test_loss)

    draw_fig(avg_train_loss, avg_test_loss, 'loss', minposs=minposs, path=path_name, idx=idx)
    draw_fig(avg_train_acc, avg_test_acc, 'acc', minposs=minposs, path=path_name, idx=idx)

'''

    for idx in enumerate(test_dataloader['v1']):
        print(idx[1][1])
        print("++++++++++++++1")

    for idx in enumerate(test_dataloader['v2']):
        print(idx[1][1])
        print("++++++++++++++2")
    for idx in enumerate(test_dataloader['v3']):
        print(idx[1][1])

